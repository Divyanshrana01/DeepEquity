#!/usr/bin/env bash
#
# Gets the demo into a state where it can actually be shown.
#
# Two separate jobs, and they're separate on purpose. Ingesting filings is free: it's SEC
# downloads and a local embedding model, no LLM involved. Warming the cache is not free,
# it runs real research passes, and Groq's free tier allows 200,000 tokens a day against
# roughly 30,000 per cold run. So ingestion runs by default and warming is opt-in, because
# accidentally spending the day's whole allowance on setup is an easy mistake to make once.
#
#   ./scripts/demo_setup.sh                 ingest the filings only
#   ./scripts/demo_setup.sh --warm          ingest, then run each ticker once
#   TICKERS="AAPL MSFT" ./scripts/demo_setup.sh --warm

set -euo pipefail

API="${API:-http://localhost:8000}"
TICKERS="${TICKERS:-AAPL MSFT NVDA}"
WARM=false
[[ "${1:-}" == "--warm" ]] && WARM=true

# The secret lives in .env and nowhere else. Read it rather than asking for it again.
if [[ -f .env ]]; then
  CLIENT_ID=$(grep -E '^DEMO_CLIENT_ID=' .env | cut -d= -f2- || echo demo)
  CLIENT_SECRET=$(grep -E '^DEMO_CLIENT_SECRET=' .env | cut -d= -f2- || echo "")
fi

if [[ -z "${CLIENT_SECRET:-}" ]]; then
  echo "DEMO_CLIENT_SECRET is not set in .env. The token endpoint stays off without it." >&2
  exit 1
fi

echo "==> getting a token"
TOKEN=$(curl -sf -X POST "$API/token" \
  -H 'Content-Type: application/json' \
  -d "{\"client_id\":\"$CLIENT_ID\",\"client_secret\":\"$CLIENT_SECRET\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

auth=(-H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json')

echo "==> requesting filings"
for ticker in $TICKERS; do
  response=$(curl -s -X POST "$API/ingest" "${auth[@]}" \
    -d "{\"ticker\":\"$ticker\",\"form_type\":\"10-K\"}")
  echo "    $ticker  $(echo "$response" | head -c 160)"
done

echo "==> waiting for the worker to finish embedding"
# Polling the corpus rather than each document, because what actually matters is whether a
# ticker is searchable, not which pipeline stage it happens to be sitting in.
for _ in $(seq 1 60); do
  ready=$(curl -s "$API/corpus" "${auth[@]}" \
    | python3 -c 'import json,sys; print(" ".join(t["ticker"] for t in json.load(sys.stdin)["tickers"]))')
  missing=""
  for ticker in $TICKERS; do
    [[ " $ready " == *" $ticker "* ]] || missing="$missing $ticker"
  done
  if [[ -z "$missing" ]]; then
    echo "    all ready: $ready"
    break
  fi
  echo "    still waiting for:$missing"
  sleep 10
done

curl -s "$API/corpus" "${auth[@]}" | python3 -m json.tool

if [[ "$WARM" == false ]]; then
  echo
  echo "Filings are in. Run again with --warm to pre-run each ticker so the demo replays"
  echo "from cache instead of taking four minutes on the first click."
  exit 0
fi

echo "==> warming the cache (a cold run is roughly 30k tokens, the daily cap is 200k)"
for ticker in $TICKERS; do
  echo "    $ticker ..."
  start=$(date +%s)
  result=$(curl -s -X POST "$API/research/sync" "${auth[@]}" -d "{\"ticker\":\"$ticker\"}")
  echo "$result" | python3 -c "
import json, sys
run = json.load(sys.stdin)
if run['status'] != 'complete':
    print('      failed:', (run.get('error') or 'no reason given')[:200])
else:
    print(f\"      done in $(( $(date +%s) - start ))s, \"
          f\"{run['tokens_used']:,} tokens, \\\${run['cost_usd']:.6f}\")
"
done

echo
echo "==> cache state"
curl -s "$API/stats" "${auth[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache"])'

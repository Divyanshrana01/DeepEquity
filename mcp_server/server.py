from fastapi import FastAPI

#placeholder for the custom mcp server. phase 2 turns this into a real mcp server exposing
#four tools: fetch_filing, fetch_transcript, fetch_price_history, fetch_news. for now it's
#just a container with a health check, so docker-compose already has the right shape and
#networking before the real tools exist
app = FastAPI(title="deepequity-mcp-server")


#just proves the container is up, same reasoning as the api's /health
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

from __future__ import annotations

import re
from dataclasses import dataclass


#One test question plus the rule that decides which passages count as correct answers.
#
#How relevance is decided matters more than anything else here. If we said "a chunk is
#relevant when it contains the words from the question", then keyword search would score
#nearly perfectly by definition and the whole measurement would be circular, it would be
#grading the search on the thing it does rather than on whether it found the right text.
#
#So every rule below is written with words that do NOT appear in its own question. The
#question asks about "disruption to production", the rule looks for suppliers and
#manufacturing. That keeps the judgement about the subject matter rather than about
#string overlap, and it gives keyword and vector search an equally fair shot.
@dataclass
class GoldenQuery:
    query_id: str
    query: str
    #a chunk counts as relevant when it matches must_match and doesn't match exclude
    must_match: list[str]
    #some rules need any-of rather than all-of, e.g. several different words for the
    #same idea
    match_mode: str = "all"
    exclude: list[str] | None = None
    #what a person would say the question is really asking, kept so the rule can be
    #argued with later rather than taken on faith
    intent: str = ""

    #decides whether one passage answers this question. lowercase everywhere so the
    #judgement doesn't hinge on how the filing happened to capitalise something.
    def is_relevant(self, text: str) -> bool:
        haystack = text.lower()

        if self.exclude and any(term.lower() in haystack for term in self.exclude):
            return False

        hits = [bool(re.search(term.lower(), haystack)) for term in self.must_match]
        return all(hits) if self.match_mode == "all" else any(hits)


#The query set. Hand written against the Apple 10-K that's actually ingested, covering a
#mix of question styles on purpose:
#
#  - semantic:  worded nothing like the filing, so only meaning-based search can find it
#  - factual:   asks for a specific number or table
#  - keyword:   names an exact term the filing definitely uses
#  - multi:     the answer is spread across several passages
#
#The mix matters. A set of only semantic questions would make vector search look perfect
#and keyword search useless, and a set of only exact-term questions would do the reverse.
GOLDEN_QUERIES: list[GoldenQuery] = [
    # --- semantic: question wording deliberately unlike the filing's wording ----------
    GoldenQuery(
        query_id="sem-01",
        query="What could go wrong if a key production partner had problems?",
        intent="supplier and manufacturing concentration risk",
        must_match=[r"suppl(y|ier|iers)", r"manufactur"],
        exclude=["water supply"],
    ),
    GoldenQuery(
        query_id="sem-02",
        query="How might rivals in the market hurt the business?",
        intent="competition risk",
        must_match=[r"competit"],
    ),
    GoldenQuery(
        query_id="sem-03",
        query="What happens if the company is taken to court?",
        intent="litigation and legal proceedings",
        must_match=[r"litigat|legal proceeding|lawsuit|claims"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="sem-04",
        query="Is the company vulnerable to hackers or data breaches?",
        intent="cybersecurity and information security risk",
        must_match=[r"cybersecurity|security incident|unauthorized access|malicious"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="sem-05",
        query="How does money moving between currencies affect results?",
        intent="foreign exchange exposure",
        must_match=[r"foreign (exchange|currency)"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="sem-06",
        query="What does the company give back to shareholders?",
        intent="dividends and share buybacks",
        must_match=[r"dividend|repurchase"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="sem-07",
        query="How reliant is the business on one part of the world for making things?",
        intent="geographic concentration of manufacturing",
        must_match=[r"outside the u\.s\.|china|asia"],
        match_mode="any",
        exclude=["net sales"],
    ),
    GoldenQuery(
        query_id="sem-08",
        query="Does the company protect its inventions and brand?",
        intent="intellectual property",
        must_match=[r"intellectual property|patent|trademark"],
        match_mode="any",
    ),
    # --- factual: a specific figure or table -----------------------------------------
    GoldenQuery(
        query_id="fact-01",
        query="What was the effective tax rate?",
        intent="the effective tax rate figure",
        must_match=[r"effective tax rate"],
    ),
    GoldenQuery(
        query_id="fact-02",
        query="How much was spent on research and development?",
        intent="R&D expense",
        must_match=[r"research and development"],
    ),
    GoldenQuery(
        query_id="fact-03",
        query="What were total net sales?",
        intent="total net sales figure",
        must_match=[r"total net sales"],
    ),
    GoldenQuery(
        query_id="fact-04",
        query="How many people does the company employ?",
        intent="headcount",
        must_match=[r"employee|headcount|full-time"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="fact-05",
        query="What is the company's exposure to interest rate changes?",
        intent="interest rate risk",
        must_match=[r"interest rate"],
    ),
    # --- keyword: names an exact term the filing uses --------------------------------
    GoldenQuery(
        query_id="kw-01",
        query="Item 1A Risk Factors",
        intent="the risk factors section",
        must_match=[r"risk factor"],
    ),
    GoldenQuery(
        query_id="kw-02",
        query="iPhone net sales",
        intent="iPhone revenue line",
        must_match=[r"iphone"],
    ),
    GoldenQuery(
        query_id="kw-03",
        query="Services segment performance",
        intent="the Services business",
        must_match=[r"services"],
    ),
    GoldenQuery(
        query_id="kw-04",
        query="share repurchase program",
        intent="buyback programme",
        must_match=[r"repurchase"],
    ),
    GoldenQuery(
        query_id="kw-05",
        query="inventory",
        intent="inventory accounting and risk",
        must_match=[r"inventor"],
    ),
    # --- multi: answer spread across several passages --------------------------------
    GoldenQuery(
        query_id="multi-01",
        query="What regulatory pressures does the business face around the world?",
        intent="regulation across jurisdictions",
        must_match=[r"regulat"],
    ),
    GoldenQuery(
        query_id="multi-02",
        query="How does the company handle its workforce and company culture?",
        intent="human capital",
        must_match=[r"employee|workforce|human capital"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="multi-03",
        query="What are the risks around new product launches?",
        intent="product transition risk",
        must_match=[r"new product|product transition|introduc"],
        match_mode="any",
    ),
    GoldenQuery(
        query_id="multi-04",
        query="How is revenue split across different regions?",
        intent="geographic segment reporting",
        must_match=[r"americas|europe|greater china|japan|rest of asia"],
        match_mode="any",
    ),
]

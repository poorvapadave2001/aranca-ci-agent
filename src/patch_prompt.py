import re, sys
p = "src/report.py"
s = open(p, encoding="utf-8").read()
NEW = """cat (pick ONE; judge by what the firm itself did or published):
M&A = the firm, its owner or portfolio acquires, merges, is acquired, divests, invests or raises funding. NOT M&A: a page or article about M&A advice, fees or trends, commentary on someone else's deal, stock-price pages.
Deal = the firm wins a client/mandate/project, signs a partnership or alliance, or appoints a senior person. NOT Deal: market reports, forecasts or guides about a market or project.
Business Update = the firm launches a service, opens an office, enters a market, hires, wins an award, reports growth.
Digital Presence = the firm's own blog, report, webinar or website page showing what it sells, its methods/tools/capabilities, or the markets and clients it targets (including how-to and capability articles).
Noise = about a different organisation, firm mentioned in passing, job ad, stock/ticker page, empty or generic page, or macro/market commentary that says nothing about the firm's own offering.
If cat is not Noise, rel must be at least 1.
"""
a = s.index("cat: Business Update")
b = s.index("If the item is about a different organisation")
if "NOT M&A" in s:
    sys.exit("already patched")
s = s[:a] + NEW + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("report.py prompt patched")
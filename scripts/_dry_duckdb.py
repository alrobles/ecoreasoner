import duckdb, time
t0 = time.time()
con = duckdb.connect()
con.execute("INSTALL sqlite_scanner; LOAD sqlite_scanner;")
n = con.execute(
    "SELECT COUNT(*) FROM sqlite_scan('/beegfs/a474r867/litdump/pubmed/pubmed_full.db','articles')"
).fetchone()[0]
yrs = con.execute(
    "SELECT MIN(CAST(year AS INTEGER)), MAX(CAST(year AS INTEGER)) "
    "FROM sqlite_scan('/beegfs/a474r867/litdump/pubmed/pubmed_full.db','articles')"
).fetchone()
print(f"rows={n} years={yrs[0]}-{yrs[1]} elapsed={round(time.time()-t0,1)}s")
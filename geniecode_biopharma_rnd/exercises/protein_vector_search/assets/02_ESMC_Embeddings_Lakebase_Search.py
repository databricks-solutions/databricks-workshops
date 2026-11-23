# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # ESMC embeddings + Lakebase Search over OAS heavy chains
# MAGIC Step 2 of the protein vector search exercise. Input: `yen.proteinsearch.oas_unique_min3` (26,013 unique VH sequences with at least 3 occurrences, built in `01_OAS_Ingestion_QC`).
# MAGIC
# MAGIC 1. Embed every sequence with the Genesis Workbench **ESMC** serving endpoint, resolved by slug at runtime, and write the results to a Delta table.
# MAGIC 2. Load the embeddings into a Lakebase Postgres table and build a **Lakebase Search** ANN index on them (`lakebase_vector` extension, `lakebase_ann` index, cosine distance).
# MAGIC 3. Run similarity searches by `seq_id` and by new sequence, then check ANN recall against exact brute-force search.
# MAGIC
# MAGIC **Before you run:** set `LAKEBASE_PROJECT_ID` in the Config cell. Then turn on **Settings -> Lakebase Search** for that project in the Lakebase UI. This can't be undone and restarts the project's computes. If it's off, `CREATE EXTENSION lakebase_vector` fails with *must be loaded via shared_preload_libraries*.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q "databricks-sdk>=0.118.0" "psycopg[binary]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
CATALOG, SCHEMA = "yen", "proteinsearch"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.oas_unique_min3"

# Embedding model, resolved by slug against the deployed GWB serving endpoints
EMBED_MODEL_SLUG = "esmc"
FALLBACK_SLUG = "esm2_embeddings"            # e.g. "esm2_embeddings" to fall back to ESM-2 if ESMC isn't deployed; empty = fail loud
BATCH_SIZE, N_THREADS = 64, 8

# Lakebase. REQUIRED: a project you hold a Postgres role in, with Settings -> Lakebase Search turned on
LAKEBASE_PROJECT_ID = "lakebase-for-students"      # e.g. "oas-antibody-search" or a project you own
CREATE_PROJECT_IF_MISSING = False
LAKEBASE_BRANCH = "production"
PG_DATABASE = "databricks_postgres"
PG_SCHEMA = "proteinsearch"

# COMMAND ----------

# DBTITLE 1,Resolve embedding endpoint and probe output
import numpy as np, pandas as pd, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

w = WorkspaceClient(config=Config(http_timeout_seconds=900))
_ALL_EPS = [e.name for e in w.serving_endpoints.list()]

def resolve_ep(slug):
    cands = [n for n in _ALL_EPS if slug in n]
    stem = lambda n: n[:-9] if n.endswith("_endpoint") else n
    return ([n for n in cands if stem(n).endswith(slug)] or cands or [None])[0]

EMBED_EP = resolve_ep(EMBED_MODEL_SLUG)
MODEL_TAG = EMBED_MODEL_SLUG
if EMBED_EP is None and FALLBACK_SLUG:
    EMBED_EP, MODEL_TAG = resolve_ep(FALLBACK_SLUG), FALLBACK_SLUG.split("_")[0]
    print(f"WARNING: no '{EMBED_MODEL_SLUG}' endpoint; falling back to {EMBED_EP}")
if EMBED_EP is None:
    raise RuntimeError(f"No serving endpoint matches '{EMBED_MODEL_SLUG}'. GWB endpoints: "
                       f"{sorted(n for n in _ALL_EPS if n.startswith('gwb_'))}. Deploy or warm ESMC, or set FALLBACK_SLUG.")

def _to_vectors(preds):
    """Normalise endpoint output to a list of float vectors (bare list, {'predictions':..}, or [{'embedding':..}])."""
    if isinstance(preds, dict):
        preds = preds.get("predictions", preds.get("embeddings", preds))
    out = []
    for p in preds:
        if isinstance(p, dict):
            p = p.get("embedding", p.get("embeddings", next(iter(p.values()))))
        out.append(np.asarray(p, dtype=np.float32).reshape(-1))
    return out

def embed(seqs, retries=4):
    for i in range(retries):
        try:
            vecs = _to_vectors(w.serving_endpoints.query(name=EMBED_EP, inputs=list(seqs)).predictions)
            assert len(vecs) == len(seqs), f"got {len(vecs)} vectors for {len(seqs)} sequences"
            return vecs
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(10 * 2 ** i)

EMB_TABLE = f"{CATALOG}.{SCHEMA}.oas_unique_min3_{MODEL_TAG}_embeddings"
t = time.time()
probe = embed(["EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"])
EMB_DIM = probe[0].shape[0]
print(f"endpoint={EMBED_EP}  dim={EMB_DIM}  probe {time.time()-t:.1f}s  -> {EMB_TABLE}")

# COMMAND ----------

# DBTITLE 1,Embed sequences -> Delta (resumable)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import functions as F, types as T

emb_schema = T.StructType([
    T.StructField("seq_id", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
    T.StructField("model_endpoint", T.StringType()),
])

src = spark.table(SOURCE_TABLE).select("seq_id", "sequence_alignment_aa")
todo = src.join(spark.table(EMB_TABLE).select("seq_id"), "seq_id", "left_anti") if spark.catalog.tableExists(EMB_TABLE) else src
todo_pdf = todo.toPandas()
print(f"{len(todo_pdf):,} sequences to embed")

batches = [todo_pdf.iloc[i:i + BATCH_SIZE] for i in range(0, len(todo_pdf), BATCH_SIZE)]
buf, t0 = [], time.time()

def _flush(rows):
    if rows:
        spark.createDataFrame(rows, emb_schema).write.mode("append").saveAsTable(EMB_TABLE)

with ThreadPoolExecutor(N_THREADS) as ex:
    futs = {ex.submit(embed, b.sequence_alignment_aa.tolist()): b for b in batches}
    for n, f in enumerate(as_completed(futs), 1):
        b = futs[f]
        buf += [(sid, v.tolist(), EMBED_EP) for sid, v in zip(b.seq_id, f.result())]
        if len(buf) >= 4000:        # checkpoint so a failure doesn't lose finished work
            _flush(buf); buf = []
        if n % 50 == 0:
            print(f"{n}/{len(batches)} batches  {time.time()-t0:.0f}s")
_flush(buf)
print(f"done in {time.time()-t0:.0f}s")

# COMMAND ----------

# DBTITLE 1,Validate embeddings table
emb = spark.table(EMB_TABLE)
display(emb.agg(
    F.count("*").alias("rows"),
    F.countDistinct("seq_id").alias("distinct_seq_ids"),
    F.min(F.size("embedding")).alias("min_dim"),
    F.max(F.size("embedding")).alias("max_dim"),
    F.sum(F.when(F.exists("embedding", lambda x: F.isnan(x)), 1).otherwise(0)).alias("rows_with_nan"),
))
n_src = spark.table(SOURCE_TABLE).count()
assert emb.count() == emb.select("seq_id").distinct().count() == n_src, "embeddings table must have exactly one row per source sequence"

# COMMAND ----------

# DBTITLE 1,Lakebase project and connection helper
import psycopg
from databricks.sdk.errors import NotFound
from databricks.sdk.service.postgres import Project, ProjectSpec, EndpointType

assert LAKEBASE_PROJECT_ID, "Set LAKEBASE_PROJECT_ID in the Config cell before running the Lakebase cells"
PROJECT = f"projects/{LAKEBASE_PROJECT_ID}"
try:
    w.postgres.get_project(name=PROJECT)
except NotFound:
    if not CREATE_PROJECT_IF_MISSING:
        raise RuntimeError(f"Lakebase project {PROJECT} not found (or no access). Set CREATE_PROJECT_IF_MISSING=True to create it.")
    w.postgres.create_project(project=Project(spec=ProjectSpec(display_name=LAKEBASE_PROJECT_ID, pg_version=17)),
                              project_id=LAKEBASE_PROJECT_ID).wait()
    print(f"Created {PROJECT}. Turn on Settings -> Lakebase Search in the Lakebase UI before continuing.")

PG_EP = next(e for e in w.postgres.list_endpoints(parent=f"{PROJECT}/branches/{LAKEBASE_BRANCH}")
             if e.status.endpoint_type == EndpointType.ENDPOINT_TYPE_READ_WRITE)
PG_HOST, PG_USER = PG_EP.status.hosts.host, w.current_user.me().user_name

def pg(autocommit=False):
    """New connection with a fresh OAuth token (tokens expire after 1h)."""
    tok = w.postgres.generate_database_credential(endpoint=PG_EP.name).token
    return psycopg.connect(host=PG_HOST, dbname=PG_DATABASE, user=PG_USER, password=tok,
                           sslmode="require", autocommit=autocommit, connect_timeout=60)

with pg() as c:
    print(c.execute("select version()").fetchone()[0])

# COMMAND ----------

# DBTITLE 1,Enable Lakebase Search extension
# No CREATE privilege on the shared default database -> use a database we own (role has CREATEDB)
with pg(autocommit=True) as c:
    if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_SCHEMA,)).fetchone():
        c.execute(f'CREATE DATABASE "{PG_SCHEMA}"')
PG_DATABASE = PG_SCHEMA   # pg() reads this global, so all later cells connect here

try:
    with pg(autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")   # CASCADE pulls in pgvector
        print(c.execute("select extname, extversion from pg_extension where extname in ('vector','lakebase_vector')").fetchall())
except psycopg.Error as e:
    raise RuntimeError("lakebase_vector unavailable. Turn on Settings -> Lakebase Search for this Lakebase "
                       "project in the UI (can't be undone; restarts computes), then re-run.") from e

# COMMAND ----------

# DBTITLE 1,Load embeddings + metadata into Postgres
PG_TABLE = f"{PG_SCHEMA}.oas_{MODEL_TAG}_embeddings"

load_pdf = (spark.table(EMB_TABLE).join(spark.table(SOURCE_TABLE), "seq_id")
            .select("seq_id", "sequence_alignment_aa", "occurrences", "cdr3_aa", "cdr3_aa_length",
                    "aa_length", "v_call", "d_call", "j_call", "embedding")).toPandas()

t0 = time.time()
with pg() as c:
    c.execute(f"CREATE SCHEMA IF NOT EXISTS {PG_SCHEMA}")
    c.execute(f"DROP TABLE IF EXISTS {PG_TABLE}")
    c.execute(f"""CREATE TABLE {PG_TABLE} (
        seq_id text PRIMARY KEY, sequence text NOT NULL, occurrences bigint, cdr3_aa text,
        cdr3_aa_length int, aa_length int, v_call text, d_call text, j_call text,
        embedding vector({EMB_DIM}) NOT NULL)""")
    cols = list(load_pdf.columns)
    with c.cursor().copy(f"COPY {PG_TABLE} ({', '.join(cols).replace('sequence_alignment_aa', 'sequence')}) FROM STDIN") as cp:
        for r in load_pdf.itertuples(index=False):
            vec = "[" + ",".join(f"{x:.7g}" for x in r.embedding) + "]"
            meta = [None if pd.isna(x) else (x.item() if hasattr(x, "item") else x) for x in r[:-1]]  # numpy -> python
            cp.write_row((*meta, vec))
    n = c.execute(f"SELECT count(*) FROM {PG_TABLE}").fetchone()[0]
print(f"loaded {n:,} rows into {PG_TABLE} in {time.time()-t0:.0f}s")
assert n == len(load_pdf)

# COMMAND ----------

# DBTITLE 1,Build lakebase_ann index
PG_INDEX = f"oas_{MODEL_TAG}_embeddings_ann"
t0 = time.time()
with pg(autocommit=True) as c:
    c.execute(f"DROP INDEX IF EXISTS {PG_SCHEMA}.{PG_INDEX}")
    c.execute(f"CREATE INDEX {PG_INDEX} ON {PG_TABLE} USING lakebase_ann (embedding vector_cosine_ops)")
    c.execute(f"ANALYZE {PG_TABLE}")
    info = c.execute("SELECT lakebase_ann_index_info(%s::regclass)", (f"{PG_SCHEMA}.{PG_INDEX}",)).fetchone()[0]
print(f"index built in {time.time()-t0:.1f}s\n{info}")

# COMMAND ----------

# DBTITLE 1,Similarity search helpers
def _vec_literal(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"

SEARCH_COLS = "seq_id, cdr3_aa, v_call, j_call, occurrences, aa_length"

def search_by_vector(vec, k=10, exclude_seq_id=None, probes=None):
    """ANN top-k by cosine similarity (1 - cosine distance). probes=None keeps the index default ('auto')."""
    lit = _vec_literal(vec)
    with pg() as c:
        if probes is not None:   # shape must match lists in lakebase_ann_index_info
            c.execute("SELECT set_config('lakebase_ann.probes', %s, true)", (probes,))
        rows = c.execute(f"""
            SELECT {SEARCH_COLS}, 1 - (embedding <=> %(q)s::vector) AS cosine_sim
            FROM {PG_TABLE}
            WHERE %(ex)s::text IS NULL OR seq_id <> %(ex)s::text
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s""", {"q": lit, "ex": exclude_seq_id, "k": k})
        cols = [d.name for d in rows.description]
        return pd.DataFrame(rows.fetchall(), columns=cols)

def search_by_sequence(seq, k=10):
    """Embed a new sequence with the same endpoint, then search."""
    return search_by_vector(embed([seq])[0], k)

def search_by_seq_id(seq_id, k=10):
    """Nearest neighbours of a sequence already in the index (itself excluded)."""
    with pg() as c:
        v = c.execute(f"SELECT embedding::text FROM {PG_TABLE} WHERE seq_id = %s", (seq_id,)).fetchone()
    if v is None: raise KeyError(seq_id)
    return search_by_vector(np.array(v[0].strip("[]").split(","), dtype=np.float32), k, exclude_seq_id=seq_id)

# COMMAND ----------

# DBTITLE 1,Similarity searches
top = spark.table(SOURCE_TABLE).orderBy(F.desc("occurrences")).select("seq_id", "sequence_alignment_aa", "cdr3_aa", "v_call").first()
print(f"Query 1: most frequent clonotype {top.seq_id}  CDR3={top.cdr3_aa}  {top.v_call}")
t0 = time.time(); hits1 = search_by_seq_id(top.seq_id, k=10); print(f"{(time.time()-t0)*1000:.0f} ms")
display(hits1)

# Query 2: unseen variant, the same sequence with 3 point mutations inside its CDR3
seq = top.sequence_alignment_aa
i = seq.find(top.cdr3_aa)
mut = list(seq)
for off, aa in zip((2, 4, 6), "AGS"):
    if i >= 0 and i + off < len(seq): mut[i + off] = aa
mut = "".join(mut)
print(f"Query 2: CDR3-mutated variant ({sum(a != b for a, b in zip(seq, mut))} substitutions)")
hits2 = search_by_sequence(mut, k=10)
display(hits2)
print("parent in top-10:", top.seq_id in set(hits2.seq_id))

# COMMAND ----------

# DBTITLE 1,Check ANN recall vs exact search
# Exact cosine top-10 from the Delta embeddings (numpy) vs lakebase_ann top-10, for 25 random queries
E = np.stack(load_pdf.embedding.map(np.asarray).values).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
ids = load_pdf.seq_id.values
rng = np.random.default_rng(0)
K, recalls = 10, []
for qi in rng.choice(len(ids), 25, replace=False):
    exact = set(ids[np.argsort(-(E @ E[qi]))[:K]])
    ann = set(search_by_vector(load_pdf.embedding.iloc[qi], k=K).seq_id)
    recalls.append(len(exact & ann) / K)
print(f"recall@{K}: mean={np.mean(recalls):.3f}  min={np.min(recalls):.2f}  (each query's top hit should be itself)")

# COMMAND ----------

# DBTITLE 1,Scope
# MAGIC %md
# MAGIC ## Scope
# MAGIC High embedding cosine similarity means two sequences look alike to the language model (germline, framework, CDR content). It is **not** evidence that they bind the same antigen. Use these hits to prioritise and enrich candidates, and confirm them with structure and binding analysis or experiments. `lakebase_ann` is approximate, and the recall check above measures how often it matches exact search.
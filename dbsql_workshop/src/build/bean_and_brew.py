"""Shared setup for the Bean & Brew build notebooks. Every value comes from the bundle's build job parameters."""
import re

SCHEMA_PARAMS = ("bronze_schema", "silver_schema", "gold_schema")


def init(spark, dbutils, extra=()):
    """Read the job parameters, USE the bundle's catalog and declare the schema session variables."""
    cfg = {}
    for name in ("catalog", *SCHEMA_PARAMS, *extra):
        dbutils.widgets.text(name, "")
        cfg[name] = dbutils.widgets.get(name).strip() if name != "asset_prefix" else dbutils.widgets.get(name)
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise ValueError(f"Missing parameters {missing}: run this notebook from the bundle's build job.")
    for k in ("catalog", *SCHEMA_PARAMS):
        if not re.fullmatch(r"[A-Za-z0-9_]+", cfg[k]):
            raise ValueError(f"{k} must contain only letters, digits and underscores: {cfg[k]!r}")
    spark.sql(f"USE CATALOG `{cfg['catalog']}`")
    for k in SCHEMA_PARAMS:
        spark.sql(f"DECLARE OR REPLACE VARIABLE {k} STRING DEFAULT '{cfg[k]}'")
    print(f"Catalog {cfg['catalog']}: " + ", ".join(cfg[k] for k in SCHEMA_PARAMS))
    return cfg


def fill(text, cfg):
    """Replace the __CATALOG__/__BRONZE__/__SILVER__/__GOLD__/__LLM_ENDPOINT__ placeholders."""
    for placeholder, key in (("__CATALOG__", "catalog"), ("__BRONZE__", "bronze_schema"), ("__SILVER__", "silver_schema"),
                             ("__GOLD__", "gold_schema"), ("__LLM_ENDPOINT__", "llm_endpoint")):
        if placeholder in text:
            text = text.replace(placeholder, cfg[key])
    return text

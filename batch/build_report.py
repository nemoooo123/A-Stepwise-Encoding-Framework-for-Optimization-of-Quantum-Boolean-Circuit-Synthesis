"""Injects batch_results/dashboard_data.json into the report template and
writes the final standalone HTML report.

Usage:
    python -m batch.build_report [output_path]
"""
import os
import sys
import json

OUT_ROOT = "batch_results"
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(HERE, "report_template.html")


def main():
    dashboard_path = os.path.join(OUT_ROOT, "dashboard_data.json")
    if not os.path.exists(dashboard_path):
        print(f"Missing {dashboard_path} — run batch.postprocess first.")
        sys.exit(1)

    with open(dashboard_path, encoding="utf-8") as f:
        data = json.load(f)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    payload = json.dumps(data, separators=(",", ":"))
    html = template.replace("/*__DATA_JSON__*/null", payload)

    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_ROOT, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_path} ({len(html)/1024:.1f} KB)")


if __name__ == "__main__":
    main()

# scripts/eval.py
import argparse, json, sys, glob

def load_metrics(pattern: str):
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        print(f"No metrics file matched: {pattern}")
        sys.exit(1)
    path = candidates[-1]
    with open(path, "r") as f:
        data = json.load(f)
    print(f"🔎 Using metrics file: {path}")
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="outputs/metrics_*.json",
                   help="metrics JSON glob pattern")
    p.add_argument("--loss-threshold", type=float, default=None,
                   help="max mean_loss allowed (fail if mean_loss > threshold)")
    p.add_argument("--map-threshold", type=float, default=None,
                   help="min mAP allowed (fail if mAP < threshold)")
    args = p.parse_args()

    m = load_metrics(args.metrics)
    ok, reasons = True, []

    mean_loss = m.get("mean_loss")
    map_val   = m.get("mAP", m.get("map"))

    if args.loss_threshold is not None:
        if mean_loss is None:
            ok = False; reasons.append("mean_loss not found in metrics")
        elif mean_loss > args.loss_threshold:
            ok = False; reasons.append(f"mean_loss {mean_loss:.4f} > {args.loss_threshold:.4f}")

    if args.map_threshold is not None:
        if map_val is None:
            ok = False; reasons.append("mAP not found in metrics")
        elif map_val < args.map_threshold:
            ok = False; reasons.append(f"mAP {map_val:.4f} < {args.map_threshold:.4f}")

    if ok:
        print("EVAL GATE PASSED")
        if mean_loss is not None: print(f"  mean_loss: {mean_loss:.4f}")
        if map_val   is not None: print(f"  mAP:       {map_val:.4f}")
        sys.exit(0)
    else:
        print("EVAL GATE FAILED")
        for r in reasons:
            print("  -", r)
        sys.exit(1)

if __name__ == "__main__":
    main()

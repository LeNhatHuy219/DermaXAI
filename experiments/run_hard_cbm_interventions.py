# Chạy đánh giá can thiệp concept đa seed cho mô hình Hard CBM.


from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hard_cbm_interventions_common import run_cli


# Gọi pipeline intervention nhiều seeds và báo cáo chung, cố định variant ST.
def main(argv=None):
    return run_cli("straight_through", argv)


if __name__ == "__main__":
    main()

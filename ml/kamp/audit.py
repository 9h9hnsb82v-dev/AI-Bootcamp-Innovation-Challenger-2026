"""KAMP ID 4 「사출성형기 AI」 원본 감사.

- PassOrFail 원본(Y/N)과 파생본(0/1) 매핑을 `_id`로 직접 대조
- 제품(PART_NAME/PART_NO)별 행 수·불량 수, 중복 레코드(동일 _id / 동일 센서벡터) 확인
- 7종 중 CN7_LH/CN7_RH/RG3_LH/RG3_RH 와 원본 PART_NAME 매핑 후보 출력

실행: .venv/Scripts/python -m ml.kamp.audit
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data/raw/kamp/04. Dataset_Molding.zip"
OUT = ROOT / "reports/audit/kamp_id4_audit.json"
SENSOR_EXCLUDE = {"_id", "TimeStamp", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME", "PART_NO",
                  "EQUIP_CD", "EQUIP_NAME", "PassOrFail", "Reason", "Unnamed: 0"}


def read_member(name: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(RAW_ZIP) as z, z.open(f"dataset/{name}") as f:
        return pd.read_csv(f, encoding="utf-8-sig", **kw)


def load_labeled() -> pd.DataFrame:
    df = read_member("labeled_data.csv", dtype={"PART_FACT_SERIAL": str})
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    return df


def main() -> dict:
    rep: dict = {"source_zip": str(RAW_ZIP.relative_to(ROOT))}
    with zipfile.ZipFile(RAW_ZIP) as z:
        rep["members"] = {i.filename: i.file_size for i in z.infolist()}

    lab = load_labeled()
    rep["labeled_data"] = {
        "shape": list(lab.shape),
        "PassOrFail_values": lab["PassOrFail"].value_counts().to_dict(),
        "Reason_values": lab["Reason"].value_counts().head(20).to_dict(),
        "time_range": [str(lab.TimeStamp.min()), str(lab.TimeStamp.max())],
        "dup__id": int(lab["_id"].duplicated().sum()),
        "dup_exact_rows_excl_id": int(lab.drop(columns=["_id"]).duplicated().sum()),
    }
    sensors = [c for c in lab.columns if c not in SENSOR_EXCLUDE]
    lab["_sensor_sig"] = pd.util.hash_pandas_object(lab[sensors], index=False)
    by = lab.groupby(["PART_NAME", "EQUIP_NAME"]).agg(
        rows=("_id", "size"), defects=("PassOrFail", lambda s: int((s == "N").sum())),
        unique_sensor_vectors=("_sensor_sig", "nunique"),
        unique_serial_per_day=("PART_FACT_SERIAL", "nunique"),
        first=("TimeStamp", "min"), last=("TimeStamp", "max"),
        days=("TimeStamp", lambda s: int(s.dt.date.nunique())))
    rep["labeled_by_part"] = {f"{a} | {b}": {k: (str(v) if k in ("first", "last") else int(v))
                                             for k, v in r.items()} for (a, b), r in by.iterrows()}
    dsig = lab[lab.PassOrFail == "N"]
    rep["defect_rows_sharing_sensor_vector"] = int(dsig["_sensor_sig"].duplicated(keep=False).sum())
    # 같은 센서벡터에 양품/불량이 섞이는가
    mixed = lab.groupby("_sensor_sig")["PassOrFail"].nunique()
    rep["sensor_vectors_with_mixed_labels"] = int((mixed > 1).sum())

    # 파생본 매핑 검증
    ml = read_member("moldset_labeled.csv", dtype={"PART_FACT_SERIAL": str})
    m = lab[["_id", "PassOrFail"]].merge(ml[["_id", "PassOrFail", "PART_NO", "PART_NAME"]],
                                         on="_id", suffixes=("_orig", "_derived"))
    rep["moldset_labeled"] = {
        "shape": list(ml.shape), "joined_on__id": int(len(m)),
        "crosstab_orig_vs_derived": pd.crosstab(m.PassOrFail_orig, m.PassOrFail_derived)
        .to_dict(),
        "part_no_by_name": ml.groupby("PART_NAME")["PART_NO"].unique().apply(list).to_dict(),
    }
    for name in ["moldset_labeled_cn7.csv", "moldset_labeled_rg3.csv", "supervised_label_cn7.csv"]:
        d = read_member(name)
        rep[name] = {"shape": list(d.shape), "PassOrFail": d["PassOrFail"].value_counts().to_dict(),
                     "note": "표준화(스케일링) 완료 파생본" if "moldset" in name else "원 스케일 파생본"}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
    print("written", OUT)

"""将沪深300成分股权重CSV转换为qlib的weights_day.txt格式。

源CSV格式:  "日期","代码","权重"
             20230630,600519.SH,5.621
目标TXT格式: date,instrument,weight
             2017-03-31,SH600000,0.004450498966335724
"""

import csv

SRC = "/home/fei/workspace/qlib/me/IC_mul/results/qlib_data/沪深300成分股权重_2017-2026半年度.csv"
DST = "/home/fei/workspace/qlib/me/IC_mul/data/weights_day.txt"


def convert_code(code: str) -> str:
    """600519.SH -> SH600519 ; 000858.SZ -> SZ000858"""
    code = code.strip()
    if code.endswith(".SH"):
        return "SH" + code[:-3]
    if code.endswith(".SZ"):
        return "SZ" + code[:-3]
    return code


def convert_date(date: str) -> str:
    """20230630 -> 2023-06-30"""
    date = date.strip()
    return f"{date[0:4]}-{date[4:6]}-{date[6:8]}"


def main():
    rows = []
    # 自动探测编码：CSV可能是 utf-8-sig 或 GBK/GB18030
    raw = open(SRC, "rb").read()
    for enc in ("utf-8-sig", "gb18030", "gbk"):
        try:
            raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"无法解码文件 {SRC}")
    with open(SRC, "r", encoding=enc, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if not r or len(r) < 3:
                continue
            date = convert_date(r[0])
            instrument = convert_code(r[1])
            # CSV权重为百分比(各日期合计~100)，qlib weights_day.txt 需为比例(合计~1)
            weight = float(r[2]) / 100.0
            rows.append((date, instrument, weight))

    # 按日期、代码排序，保证输出稳定
    rows.sort(key=lambda x: (x[0], x[1]))

    with open(DST, "w", encoding="utf-8", newline="") as f:
        f.write("date,instrument,weight\n")
        for date, instrument, weight in rows:
            f.write(f"{date},{instrument},{weight}\n")

    print(f"已写入 {len(rows)} 行 -> {DST}")


if __name__ == "__main__":
    main()

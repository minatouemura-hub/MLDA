import random
from pathlib import Path

import pandas as pd


def load_100_data(cap_path: Path) -> pd.DataFrame:
    """
    cap_path: キャプションファイルのパス（1行につき「<ファイル名> <キャプション>」形式）
    img_dir_path: 画像ディレクトリのパス
    戻り値: ランダムに選んだ100件を 'image','caption' 列のDataFrameで返す
    """
    # キャプションファイルを読み込む
    with open(cap_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # ランダムに100件選択（行数が100未満なら全件）
    sample_lines = random.sample(lines, min(100, len(lines)))

    records = []
    for line in sample_lines:
        # "<filename>.jpg キャプション文..." を分割
        parts = line.split(sep=",", maxsplit=1)
        if len(parts) != 2:
            continue  # 形式不正行はスキップ
        fname, caption = parts
        records.append({"image": str(fname), "caption": caption})

    # DataFrame化して返す
    return pd.DataFrame(records, columns=["image", "caption"])

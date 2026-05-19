import numpy as np
import cshogi
import os
from tqdm import tqdm

INPUT_BIN = "data/Hao007.bin"
OUTPUT_TXT = "data/Hao007.txt"

def convert_psv_to_txt_for_v15():
    if not os.path.exists(INPUT_BIN):
        print(f"エラー: {INPUT_BIN} が見つかりません。")
        return

    file_size = os.path.getsize(INPUT_BIN)
    num_records = file_size // 40
    print(f"ファイルサイズ: {file_size:,} bytes")
    print(f"総レコード数 (ゴミデータ含む): {num_records:,} 局面")

    # 40バイト(PSV)として読み込み
    records = np.memmap(INPUT_BIN, dtype=cshogi.PackedSfenValue, mode='r')
    
    valid_count = 0
    error_count = 0

    # ★修正1: boardオブジェクトを事前に作成
    board = cshogi.Board()

    # ★修正2: 書き込み用ファイルをオープン
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for i in tqdm(range(num_records), desc="テキスト変換中"):
            try:
                record = records[i]
                
                # 1. 盤面と持ち駒を解凍 (PSV専用)
                board.set_psfen(record['sfen'])
                
                # 2. 正しい手数を反映 (PackedSfenValueのgamePly)
                board.move_number = int(record['gamePly'])
                
                # 3. SFEN、指し手、スコアを取得
                sfen = board.sfen()
                # PSV形式の指し手を変換
                move_val = cshogi.move16_from_psv(record['move'])
                move_usi = cshogi.move_to_usi(move_val)
                score = record['score']
                
                # 4. 保存
                f.write(f"{sfen} {move_usi} {score}\n")
                valid_count += 1
                
            except Exception:
                error_count += 1
    
    print("\n=== 変換完了 ===")
    print(f"✅ 正常に抽出された局面 (学習用) : {valid_count:,}")
    print(f"🗑️ 破棄したゴミデータ : {error_count:,}")
    print(f"出力先: {OUTPUT_TXT}")

if __name__ == "__main__":
    convert_psv_to_txt_for_v15()
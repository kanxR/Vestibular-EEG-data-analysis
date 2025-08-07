import pyxdf
import os

# ★★★ あなたのXDFファイルへのパス ★★★
fname = r'your file path' # 先ほどと同じファイルパス

if not os.path.exists(fname):
    print(f"Error: File not found: {fname}")
else:
    try:
        streams, header = pyxdf.load_xdf(fname)
        print(f"'{fname}' の中身を詳細に調査します...")
        print("-" * 50)

        # 全てのストリームをループして情報を表示
        for i, stream in enumerate(streams):
            info = stream['info']
            name = info.get('name', ['N/A'])[0]
            stype = info.get('type', ['N/A'])[0]
            n_channels = int(info.get('channel_count', [0])[0])
            
            print(f"\n【ストリーム {i+1}】")
            print(f"  Name: '{name}'")
            print(f"  Type: '{stype}'")
            print(f"  チャンネル数: {n_channels}")

            # チャンネル名の詳細を表示
            try:
                # 'desc'の中にチャンネル情報があるか試す
                channels_info = info.get('desc', [{}])[0].get('channels', [{}])[0].get('channel', [])
                if channels_info:
                    print(f"  チャンネルリスト:")
                    ch_names = [ch.get('label', ['N/A'])[0] for ch in channels_info]
                    for j, ch_name in enumerate(ch_names):
                        print(f"    {j+1}: '{ch_name}'")
                else:
                    print("  詳細なチャンネル情報は見つかりませんでした。")
            except (KeyError, IndexError):
                print("  詳細なチャンネル情報は見つかりませんでした。")
        
        print("\n" + "-" * 50)
        print("調査完了。")

    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
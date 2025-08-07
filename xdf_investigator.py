import pyxdf
import os

# ������ ���Ȃ���XDF�t�@�C���ւ̃p�X ������
fname = r'your file path' # ��قǂƓ����t�@�C���p�X

if not os.path.exists(fname):
    print(f"Error: File not found: {fname}")
else:
    try:
        streams, header = pyxdf.load_xdf(fname)
        print(f"'{fname}' �̒��g���ڍׂɒ������܂�...")
        print("-" * 50)

        # �S�ẴX�g���[�������[�v���ď���\��
        for i, stream in enumerate(streams):
            info = stream['info']
            name = info.get('name', ['N/A'])[0]
            stype = info.get('type', ['N/A'])[0]
            n_channels = int(info.get('channel_count', [0])[0])
            
            print(f"\n�y�X�g���[�� {i+1}�z")
            print(f"  Name: '{name}'")
            print(f"  Type: '{stype}'")
            print(f"  �`�����l����: {n_channels}")

            # �`�����l�����̏ڍׂ�\��
            try:
                # 'desc'�̒��Ƀ`�����l����񂪂��邩����
                channels_info = info.get('desc', [{}])[0].get('channels', [{}])[0].get('channel', [])
                if channels_info:
                    print(f"  �`�����l�����X�g:")
                    ch_names = [ch.get('label', ['N/A'])[0] for ch in channels_info]
                    for j, ch_name in enumerate(ch_names):
                        print(f"    {j+1}: '{ch_name}'")
                else:
                    print("  �ڍׂȃ`�����l�����͌�����܂���ł����B")
            except (KeyError, IndexError):
                print("  �ڍׂȃ`�����l�����͌�����܂���ł����B")
        
        print("\n" + "-" * 50)
        print("���������B")

    except Exception as e:
        print(f"�������ɃG���[���������܂���: {e}")
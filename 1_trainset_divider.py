import random
import os

# 파일 경로 설정
input_path = 'datasets/kaist-rgbt/train-all-04.txt'
output_train_path = 'datasets/kaist-rgbt/train-all-04_train_80.txt'
output_val_path = 'datasets/kaist-rgbt/train-all-04_val_20.txt'

# 파일이 존재하는지 확인
if not os.path.exists(input_path):
    print(f"Input file does not exist: {input_path}")
else:
    # 파일을 읽고 내용을 리스트로 저장
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # 리스트를 섞어서 무작위 순서로 만듦
    random.shuffle(lines)

    # 8:2 비율로 나누기
    split_index = int(0.8 * len(lines))
    train_lines = lines[split_index:]
    val_lines = lines[:split_index]

    # train.txt 파일로 저장
    try:
        with open(output_train_path, 'w') as train_file:
            train_file.writelines(train_lines)
        print(f"Train file saved successfully: {output_train_path}")
    except Exception as e:
        print(f"Error saving train file: {e}")

    # val.txt 파일로 저장
    try:
        with open(output_val_path, 'w') as val_file:
            val_file.writelines(val_lines)
        print(f"Validation file saved successfully: {output_val_path}")
    except Exception as e:
        print(f"Error saving validation file: {e}")

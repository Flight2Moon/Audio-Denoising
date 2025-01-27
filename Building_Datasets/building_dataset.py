import os
import csv
import wave
from audiomentations import Compose, AddBackgroundNoise, Normalize
import soundfile as sf

def make_dataset(
    dataset_load_paths,
    min_max_snr,
    file_extension=".wav",
    dataset_size=5,
    output_dir_path="./output_samples",
    label_path="./overlayed_data_label.csv",
    apply_normalize=True  
):
    if not dataset_load_paths or any(not os.path.exists(path) for path in dataset_load_paths):
        print("Error Code 02: Invalid dataset paths provided.")
        return

    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Output directory created: {output_dir_path}")

    
    existing_data = []
    if os.path.exists(label_path):
        with open(label_path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_data = [row for row in reader]

    
    existing_file_names = {row["file_name"] for row in existing_data}

    
    data_paths = []

    
    for directory in dataset_load_paths:
        filenames = os.listdir(directory)
        file_paths = [
            os.path.join(directory, filename)
            for filename in filenames
            if os.path.splitext(filename)[-1] == file_extension
        ]
        data_paths.append(file_paths)

    
    if len(data_paths[0]) < dataset_size:
        print("Error Code 01: Not enough files in the primary directory.")
        return

    number_of_directory = len(dataset_load_paths)
    print("Check Input Directory Audio Files Information")

    
    audio_total_information = []
    for i in range(number_of_directory):
        with wave.open(data_paths[i][0], "rb") as file:
            audio_total_information.append([
                file.getnchannels(),
                file.getsampwidth(),
                file.getframerate(),
                file.getnframes(),
                file.getnframes() / file.getframerate()
            ])

    for i in range(number_of_directory - 1):
        if audio_total_information[i] != audio_total_information[i + 1]:
            print(f"Error: Audio information mismatch between dir{i} and dir{i + 1}")
            return

    label_data = []

    
    if apply_normalize:
        normalize_transform = Compose([
            Normalize(apply_to="all", p = 1.0)  
        ])

    for i in range(dataset_size):
        main_audio, sample_rate = sf.read(data_paths[0][i])

        
        if apply_normalize:
            main_audio = normalize_transform(samples=main_audio, sample_rate=sample_rate)

        for j in range(number_of_directory - 1):
            min_snr, max_snr = min_max_snr[j]
            augment = Compose([
                AddBackgroundNoise(
                    sounds_path=data_paths[j + 1][i],
                    min_snr_db=min_snr,
                    max_snr_db=max_snr,
                    p=1.0
                )
            ])
            main_audio = augment(samples=main_audio, sample_rate=sample_rate)

        output_file_name = f"overlayed_data{i}_{number_of_directory}_toycar_AD2_paptab3c_train__toytrain_AD2_paptab3c_toytrain_target_test__toycar_AD2_paptab3c_source_test.wav"
        output_path = os.path.join(output_dir_path, output_file_name)

        
        if output_file_name not in existing_file_names:
            label_data.append({
                "file_name": output_file_name,
                "label": f"{number_of_directory}"
            })


        sf.write(output_path, main_audio, sample_rate)


    all_label_data = existing_data + label_data

    with open(label_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["file_name", "label"])
        writer.writeheader()
        writer.writerows(all_label_data)

    print("Dataset Building Complete")



def main():

  path1 = './sample_path/audio_data_path1'
  path2 = './sample_path/audio_data_path1'
  path3 = './sample_path/audio_data_path1'



    ######## Function Call ########

    make_dataset(
        [
            path1,
            path2,
            path3,
        ],
        [
            [0, 1.0],
            [0, 1.0],
            [0, 1.0],
        ],
        '.wav',
        1516,
        dataset_save_path,
        dataset_save_path + lable_file_name,
        True,
        
    )

if __name__ == '__main__': main()


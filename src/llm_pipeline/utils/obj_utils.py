import os
import csv

def csv_to_obj(csv_file_path: str, obj_folder_path: str, obj_column_index: int):
    """Writes text from each row of a specified column to a separate obj file with the same filename as the csv file into a folder
    """
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_file_path}")

    name = os.path.splitext(os.path.basename(csv_file_path))[0]  # Remove .csv extension

    with open(csv_file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader, None)  # Skip header if present

        counter = 0
        for row in reader:
            if len(row) > obj_column_index:
                file_name = f"{name}_{counter}.obj"
                create_obj_file(obj_folder_path, file_name, row[obj_column_index])
                print(f"Created {os.path.join(obj_folder_path, file_name)}")
                counter += 1


def create_obj_file(folder_path: str, file_name: str, text: str) -> bool:
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    if not file_name.endswith('.obj'):
        file_name += ".obj"

    with open(os.path.join(folder_path, file_name), "w", encoding="utf-8") as file:
        file.write(text)
    return True


if __name__=="__main__":
    pass
def filter_log_file(input_file, output_file):
    """
    Фильтрует строки из файла, удаляя те, которые начинаются с указанных префиксов.
    
    Args:
        input_file (str): Путь к исходному файлу
        output_file (str): Путь к выходному файлу
    """
    
    # Список префиксов для удаления
    prefixes_to_remove = [
        "{'loss':",
        "{'eval_loss':", 
        "{'train_runtime':",
        "  active_adapters",
        "New chosen layers:",
        "self.lora_ranks:",
        "We will use the GPU:",
        "model is",
        "trainable params",
        "[H[JWe",
        "        ",
    ]
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # Проверяем, начинается ли строка с любого из префиксов
                    if not any(line.startswith(prefix) for prefix in prefixes_to_remove):
                        outfile.write(line)
        
        print(f"Файл успешно обработан!")
        print(f"Исходный файл: {input_file}")
        print(f"Результирующий файл: {output_file}")
        
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования
if __name__ == "__main__":
    input_filename = "paper_exp_20092025.txt"  # Замените на путь к вашему файлу
    output_filename = "paper_exp_20092025_parsed.txt"  # Замените на желаемое имя выходного файла
    
    filter_log_file(input_filename, output_filename)
Исходный код проекта:

Система для моделирования метода параметризации геологических моделей и их адаптации к истории разработки

Система реализована на языке Python. Использовались следующие инструменты и библиотеки:

PyTorch, https://github.com/pytorch/examples/tree/master/fast_neural_style

OPM Flow

Ecl (Eclipse format Tools)

pyswarms

Для запуска необходимо установить гидродинамический симулятор OPM Flow (https://opm-project.org/)

Пример использования приведен в файле Main.ipynb. Получить набор реализаций можно модулем snesim.py, обучить параметризующую модель - модулем Training.ipynb

Тестовые данные для гидродинамического симулятора приведены в test_data/models/EGG (построены на основе https://data.4tu.nl/articles/dataset/The_Egg_Model_-_data_files/12707642)

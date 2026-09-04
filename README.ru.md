# Neuroacoustic Resonator

[English version](README.md)

[![CI](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml/badge.svg)](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Экспериментальное поле связанных осцилляторов, которое преобразует звук в
развивающуюся динамику, записывает её как Sound Protocol v0 и создаёт управляемый
протоколом ответ. Это не языковая модель; проект пока не демонстрирует язык,
мышление или выученные акустические понятия.

## Текущее состояние

Главный исследовательский вопрос — создают ли разные звуки воспроизводимые ответы,
не зависящие от начального seed. Парные stimulus/control ветви теперь запускаются
из одного equilibrated checkpoint. Первый pilot на трёх roots получил cross-seed
balanced accuracy `0.542` при chance `0.25` и checkpoint-clustered 95% CI
`[0.274; 0.601]`. Это перспективный pilot, а не завершённое доказательство.

Текущая цепочка:

```text
звук -> input/association/output поле -> Sound Protocol v0 -> анализ или renderer
```

## Установка

Нужны Python 3.13, [uv](https://docs.astral.sh/uv/) и
[just](https://just.systems/).

```bash
just sync
just check
just preview
```

Артефакты создаются в `experiments/` и `outputs/` и по умолчанию не коммитятся.

## Команды

`just` показывает все актуальные публичные recipes.

### Окружение и качество

| Команда | Назначение |
|---|---|
| `just sync` | Установить зафиксированное runtime/dev-окружение. |
| `just check` | Запустить Ruff, проверку форматирования, mypy и все тесты. |
| `just fmt` | Применить безопасные Ruff-исправления и форматирование. |
| `just test [аргументы pytest]` | Запустить все или выбранные тесты. |
| `just build` | Собрать Python-пакет. |
| `just hooks-install` | Установить pre-commit hook. |
| `just hooks` | Немедленно запустить все pre-commit проверки. |

### Текущее исследование

| Команда | Назначение |
|---|---|
| `just causal-run` | Запустить 30 equilibrated stimulus/control ветвей и пересобрать evidence. |
| `just causal-analyze` | Пересчитать checkpoint-aware статистику по готовым embeddings без симуляций. |

### Поле и Sound Protocol

| Команда | Назначение |
|---|---|
| `just preview` | Запустить короткую симуляцию и создать `outputs/field-preview.png`. |
| `just protocol-record [config] [steps] [output]` | Записать строгий JSONL; по умолчанию `field_only.yaml` и 128 шагов. |
| `just protocol-replay [input] [output]` | Воспроизвести JSONL без поля и создать диагностический WAV. |

### Работа со звуком

| Команда | Назначение |
|---|---|
| `just audio-devices` | Показать устройства ввода и вывода. |
| `just live` | Запустить пошаговое взаимодействие через микрофон с записью сессии. |
| `just conversation <input.wav> [output.wav]` | Пропустить WAV через поле и сохранить ответ. |

### Состояние и производительность

| Команда | Назначение |
|---|---|
| `just checkpoint [steps] [output]` | Сохранить состояние поля; по умолчанию после 10 000 шагов. |
| `just resume <checkpoint> <steps> <output>` | Продолжить симуляцию в новый checkpoint. |
| `just benchmark` | Измерить скорость поля размеров 64, 128 и 200. |

## Лицензия

[MIT](LICENSE)

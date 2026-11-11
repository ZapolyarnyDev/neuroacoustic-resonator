# Neuroacoustic Resonator

[English](README.md) · Русский · [العربية](README.ar.md) · [עברית](README.he.md)

[![CI](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml/badge.svg)](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Экспериментальное поле связанных фазовых осцилляторов с локальным метаболизмом и
пластичностью, на которое подаётся звук. Состояния поля можно записывать как
Sound Protocol v0 и озвучивать. Это не языковая модель; проект пока не
демонстрирует язык, мышление или выученные акустические понятия.

## Текущее состояние

В текущем поле входной сигнал не доходит до выходного региона. Отклик выхода на
каждый стимул совпадает с контролем тишины, а возмущение гаснет в 50–100 раз на
столбец. Прежний pilot-результат (`0.542` cross-seed balanced accuracy, тег
`v0-baseline`) получался из-за прямой подачи звука в ассоциативный регион; без неё
и поле, и поле без связей сигнала не показывают
([отчёт](experiments/reports/2026-11-06-9748330.md)).

## Установка

Нужны Python 3.13, [uv](https://docs.astral.sh/uv/) и
[just](https://just.systems/).

```bash
just sync
just check
just preview
```

Каждый recipe вызывает `uv run neuroacoustic <команда>`; полный список —
`uv run neuroacoustic --help`.

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

### Исследование

| Команда | Назначение |
|---|---|
| `just report` | Записать ключевой отчёт текущего коммита в `experiments/reports/`. |
| `just causal-run` | Запустить парный stimulus/control корпус (30 ветвей из общих equilibrated checkpoints) и пересобрать evidence. |
| `just causal-analyze` | Пересчитать checkpoint-aware output и региональную causal-статистику без симуляций. |
| `just causal-parts` | Измерить отклик с поочерёдно выключенными частями поля и с отрезанным выходным регионом. |
| `just causal-ablations` | Запустить четыре варианта equilibration на development seeds и сравнить output. |
| `just causal-ablations-report` | Пересобрать сравнение по готовым вариантам без симуляций. |

### Поле, состояние и производительность

| Команда | Назначение |
|---|---|
| `just preview` | Запустить короткую симуляцию и создать `outputs/field-preview.png`. |
| `just checkpoint [steps] [output]` | Сохранить состояние поля; по умолчанию после 10 000 шагов. |
| `just resume <checkpoint> <steps> <output>` | Продолжить симуляцию в новый checkpoint. |
| `just benchmark` | Измерить скорость поля размеров 64, 128 и 200. |

### Sound Protocol

| Команда | Назначение |
|---|---|
| `just protocol-record [config] [steps] [output]` | Записать строгий JSONL; по умолчанию `field_only.yaml` и 128 шагов. |
| `just protocol-replay [input] [output]` | Воспроизвести JSONL без поля и создать диагностический WAV. |

### Работа со звуком

| Команда | Назначение |
|---|---|
| `just audio-devices` | Показать устройства ввода и вывода. |
| `just live` | Запустить пошаговое взаимодействие через микрофон с записью сессии. |
| `just conversation <input.wav> [output.wav]` | Пропустить WAV через поле и сохранить ответ. |

## Лицензия

[MIT](LICENSE)

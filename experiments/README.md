# Experiments

Эта папка хранит локальные результаты исследовательских запусков: логи, состояния, аудио и видео.

Большие и воспроизводимые артефакты не коммитятся

## Этап 1: различимость и повторяемость

Подготовить семь сценариев robustness matrix:

```text
just stage-one-prepare
```

Запустить всю матрицу либо один сценарий:

```text
just stage-one
just stage-one-scenario baseline
just stage-one-scenario coupling-low
```

Повторно выполнить анализ уже записанных protocol streams:

```text
just stage-one-analysis
```

Собрать итоговый pass/fail gate без запуска симуляций:

```text
just stage-one-gate
```

Этап считается экспериментально завершённым только при `status: passed` в
`experiments/logs/distinguishability_stage_one_report.json`. Статусы `incomplete`
и `failed` не разрешают переход к Этапу 2.

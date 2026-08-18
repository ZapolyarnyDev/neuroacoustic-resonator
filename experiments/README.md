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

Если gate завершился со `status: failed`, снять variance и distance diagnostics без
повторного запуска симуляций:

```text
just stage-one-diagnostics
```

Команда создаёт `diagnostics.json`, таблицы и график внутри каждого scenario, а
общую сводку записывает в
`experiments/logs/distinguishability_stage_one_diagnostics.json`.

Сравнить абсолютные и seed-invariant response representations на уже записанных
Sound Protocol streams:

```text
just stage-one-representations
```

Representation выбирается по train/validation. Test читается только после выбора.

Запустить малый paired stimulus/control корпус из общих equilibrated checkpoints:

```text
just controlled-equilibration
```

Команда создаёт шесть checkpoint для трёх seed roots и двух независимых повторов,
запускает 30 ветвей и измеряет 24 активных stimulus как покадровую причинную
разность относительно silence-ветви с тем же checkpoint fingerprint.

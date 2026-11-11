# Experiments

Каталог содержит локальные protocol streams, checkpoints, WAV, таблицы, графики и
статистические отчёты. Большие воспроизводимые артефакты не коммитятся.

## Парный корпус stimulus/control

Запустить paired stimulus/control corpus из общих equilibrated checkpoints:

```text
just causal-run
```

Команда создаёт 6 checkpoints, 30 ветвей и 24 causal pairs, затем рассчитывает
absolute/causal diagnostics и checkpoint-aware evidence.

Пересчитать статистику по готовым causal embeddings без новых симуляций:

```text
just causal-analyze
```

Основные результаты:

- `experiments/logs/controlled_equilibration_baseline.json` — общая сводка;
- `experiments/controlled_equilibration/baseline/causal_evidence.json` — accuracy,
  результаты по roots/checkpoints, bootstrap и permutation;
- `experiments/controlled_equilibration/baseline/regional_causal_report.json` —
  сравнение input, assoc, output и объединённого пространственно-временного отклика;
- `experiments/controlled_equilibration/baseline/causal_diagnostics.png` — variance
  decomposition и distance distributions.

`experiments/reports/` — ключевой отчёт на коммит (`just report`), коммитится.

Модули прежних экспериментов (Stage 1, calibration, probes) остаются в пакете без
команд запуска — для воспроизводимости уже полученных результатов.

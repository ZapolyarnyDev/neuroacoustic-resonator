# Neuroacoustic Resonator

[English](README.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · עברית

[![CI](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml/badge.svg)](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div dir="rtl">

שדה ניסיוני של מתנדים פאזיים מצומדים, עם חילוף חומרים מקומי ופלסטיות, המוזן
בצליל. ניתן להקליט את מצבי השדה כזרמי Sound Protocol v0 ולהפוך אותם לצליל. זה
אינו מודל שפה, והפרויקט אינו מדגים כרגע שפה, קוגניציה או מושגים אקוסטיים נלמדים.

## מצב נוכחי

בשדה הנוכחי אות הקלט אינו מגיע לאזור הפלט. תגובת הפלט לכל גירוי זהה לבקרת
השקט, וההפרעה דועכת פי 50–100 בכל עמודה. תוצאת הפיילוט הקודמת (`0.542`
cross-seed balanced accuracy, תג `v0-baseline`) נבעה מהזנת הצליל ישירות לאזור
האסוציאטיבי; בלעדיה גם השדה וגם שדה ללא צימוד אינם מראים אות
([דוח](experiments/reports/2026-11-06-9748330.md)).

## התקנה

נדרשים Python 3.13,‏ [uv](https://docs.astral.sh/uv/) ו־[just](https://just.systems/).

</div>

```bash
just sync
just check
just preview
```

<div dir="rtl">

כל מתכון מריץ `uv run neuroacoustic <command>`; לרשימה המלאה הריצו
`uv run neuroacoustic --help`.

תוצרים נוצרים ב־`experiments/` וב־`outputs/` ואינם נכנסים למאגר כברירת מחדל.

## פקודות

הריצו `just` כדי להציג את המתכונים הציבוריים הנוכחיים.

### סביבה ואיכות

| פקודה | מטרה |
|---|---|
| `just sync` | התקנת סביבת ההרצה והפיתוח הנעולה. |
| `just check` | הרצת Ruff, בדיקת עיצוב, mypy וכל הבדיקות. |
| `just fmt` | החלת תיקוני Ruff בטוחים ועיצוב קובצי Python. |
| `just test [pytest args]` | הרצת כל הבדיקות או העברת נתיב/אפשרויות ל־pytest. |
| `just build` | בניית חבילת Python. |
| `just hooks-install` | התקנת hook של pre-commit במאגר. |
| `just hooks` | הרצה מיידית של כל בדיקות pre-commit. |

### מחקר

| פקודה | מטרה |
|---|---|
| `just report` | כתיבת הדוח המרכזי של הקומיט הנוכחי ל־`experiments/reports/`. |
| `just causal-run` | הרצת קורפוס גירוי/בקרה מזווג (30 ענפים מנקודות שמירה מאוזנות משותפות) ובנייה מחדש של הראיות. |
| `just causal-analyze` | חישוב מחדש של סטטיסטיקות הפלט והסטטיסטיקה הסיבתית האזורית ללא סימולציות. |
| `just causal-parts` | מדידת התגובה כשכל רכיב של השדה מושבת וכשאזור הפלט מנותק. |
| `just causal-ablations` | הרצת ארבע גרסאות איזון על seeds של פיתוח והשוואת תגובות הפלט. |
| `just causal-ablations-report` | בנייה מחדש של השוואת הגרסאות מתוצאות מוכנות. |

### שדה, מצב וביצועים

| פקודה | מטרה |
|---|---|
| `just preview` | הרצת סימולציה קצרה וכתיבת `outputs/field-preview.png`. |
| `just checkpoint [steps] [output]` | שמירת מצב השדה; כברירת מחדל אחרי 10,000 צעדים. |
| `just resume <checkpoint> <steps> <output>` | המשך סימולציה שמורה לנקודת שמירה חדשה. |
| `just benchmark` | מדידת מהירות צעדי השדה בגדלים 64,‏ 128 ו־200. |

### Sound Protocol

| פקודה | מטרה |
|---|---|
| `just protocol-record [config] [steps] [output]` | הקלטת זרם JSONL קפדני; כברירת מחדל `field_only.yaml` ו־128 צעדים. |
| `just protocol-replay [input] [output]` | ניגון JSONL ללא השדה והפקת קובץ WAV אבחוני. |

### אינטראקציה קולית

| פקודה | מטרה |
|---|---|
| `just audio-devices` | הצגת התקני קלט ופלט. |
| `just live` | התחלת אינטראקציה בתורות דרך המיקרופון והקלטת המפגש. |
| `just conversation <input.wav> [output.wav]` | העברת קובץ WAV מוקלט דרך השדה ושמירת התגובה. |

## רישיון

[MIT](LICENSE)

</div>

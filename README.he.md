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
השקט, וההפרעה דועכת פי 50–100 בכל עמודה. תוצאת הפיילוט הקודמת (<span dir="ltr">`0.542`</span>
cross-seed balanced accuracy, תג <span dir="ltr">`v0-baseline`</span>) נבעה מהזנת הצליל ישירות לאזור
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

כל מתכון מריץ <span dir="ltr">`uv run neuroacoustic <command>`</span>; לרשימה המלאה הריצו
<span dir="ltr">`uv run neuroacoustic --help`</span>.

תוצרים נוצרים ב־<span dir="ltr">`experiments/`</span> וב־<span dir="ltr">`outputs/`</span> ואינם נכנסים למאגר כברירת מחדל.

## פקודות

הריצו <span dir="ltr">`just`</span> כדי להציג את המתכונים הציבוריים הנוכחיים.

### סביבה ואיכות

| פקודה | מטרה |
|---|---|
| <span dir="ltr">`just sync`</span> | התקנת סביבת ההרצה והפיתוח הנעולה. |
| <span dir="ltr">`just check`</span> | הרצת Ruff, בדיקת עיצוב, mypy וכל הבדיקות. |
| <span dir="ltr">`just fmt`</span> | החלת תיקוני Ruff בטוחים ועיצוב קובצי Python. |
| <span dir="ltr">`just test [pytest args]`</span> | הרצת כל הבדיקות או העברת נתיב/אפשרויות ל־pytest. |
| <span dir="ltr">`just build`</span> | בניית חבילת Python. |
| <span dir="ltr">`just hooks-install`</span> | התקנת hook של pre-commit במאגר. |
| <span dir="ltr">`just hooks`</span> | הרצה מיידית של כל בדיקות pre-commit. |

### מחקר

| פקודה | מטרה |
|---|---|
| <span dir="ltr">`just report`</span> | כתיבת הדוח המרכזי של הקומיט הנוכחי ל־<span dir="ltr">`experiments/reports/`</span>. |
| <span dir="ltr">`just causal-run`</span> | הרצת קורפוס גירוי/בקרה מזווג (30 ענפים מנקודות שמירה מאוזנות משותפות) ובנייה מחדש של הראיות. |
| <span dir="ltr">`just causal-analyze`</span> | חישוב מחדש של סטטיסטיקות הפלט והסטטיסטיקה הסיבתית האזורית ללא סימולציות. |
| <span dir="ltr">`just causal-parts`</span> | מדידת התגובה כשכל רכיב של השדה מושבת וכשאזור הפלט מנותק. |
| <span dir="ltr">`just causal-ablations`</span> | הרצת ארבע גרסאות איזון על seeds של פיתוח והשוואת תגובות הפלט. |
| <span dir="ltr">`just causal-ablations-report`</span> | בנייה מחדש של השוואת הגרסאות מתוצאות מוכנות. |

### שדה, מצב וביצועים

| פקודה | מטרה |
|---|---|
| <span dir="ltr">`just preview`</span> | הרצת סימולציה קצרה וכתיבת <span dir="ltr">`outputs/field-preview.png`</span>. |
| <span dir="ltr">`just checkpoint [steps] [output]`</span> | שמירת מצב השדה; כברירת מחדל אחרי 10,000 צעדים. |
| <span dir="ltr">`just resume <checkpoint> <steps> <output>`</span> | המשך סימולציה שמורה לנקודת שמירה חדשה. |
| <span dir="ltr">`just benchmark`</span> | מדידת מהירות צעדי השדה בגדלים 64,‏ 128 ו־200. |

### Sound Protocol

| פקודה | מטרה |
|---|---|
| <span dir="ltr">`just protocol-record [config] [steps] [output]`</span> | הקלטת זרם JSONL קפדני; כברירת מחדל <span dir="ltr">`field_only.yaml`</span> ו־128 צעדים. |
| <span dir="ltr">`just protocol-replay [input] [output]`</span> | ניגון JSONL ללא השדה והפקת קובץ WAV אבחוני. |

### אינטראקציה קולית

| פקודה | מטרה |
|---|---|
| <span dir="ltr">`just audio-devices`</span> | הצגת התקני קלט ופלט. |
| <span dir="ltr">`just live`</span> | התחלת אינטראקציה בתורות דרך המיקרופון והקלטת המפגש. |
| <span dir="ltr">`just conversation <input.wav> [output.wav]`</span> | העברת קובץ WAV מוקלט דרך השדה ושמירת התגובה. |

## רישיון

[MIT](LICENSE)

</div>

# Neuroacoustic Resonator

[English](README.md) · [Русский](README.ru.md) · العربية · [עברית](README.he.md)

[![CI](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml/badge.svg)](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div dir="rtl">

حقل تجريبي من المذبذبات الطورية المقترنة، ذو أيض موضعي ولدونة، يُغذّى بالصوت.
يمكن تسجيل حالات الحقل كتدفقات Sound Protocol v0 وتحويلها إلى صوت. هذا ليس
نموذجًا لغويًا، ولا يُظهر المشروع حاليًا لغةً أو إدراكًا أو مفاهيم صوتية مُتعلَّمة.

## الحالة الراهنة

في الحقل الحالي لا تصل إشارة الدخل إلى منطقة الخرج. استجابة الخرج لكل مُحفِّز
مطابقة لضابط الصمت، ويخمد الاضطراب بمعامل 50–100 في كل عمود. النتيجة التجريبية
السابقة (`0.542` cross-seed balanced accuracy، الوسم `v0-baseline`) نتجت عن تغذية
الصوت مباشرةً إلى المنطقة الترابطية؛ ومن دونها لا يُظهر أيٌّ من الحقل والحقل غير
المقترن أي إشارة ([التقرير](experiments/reports/2026-11-06-9748330.md)).

## الإعداد

يتطلب Python 3.13 و[uv](https://docs.astral.sh/uv/) و[just](https://just.systems/).

</div>

```bash
just sync
just check
just preview
```

<div dir="rtl">

كل وصفة تستدعي `uv run neuroacoustic <command>`؛ للقائمة الكاملة شغّل
`uv run neuroacoustic --help`.

تُكتب المخرجات المولَّدة في `experiments/` و`outputs/` ولا تُضاف إلى المستودع افتراضيًا.

## الأوامر

شغّل `just` لعرض الوصفات العامة الحالية.

### البيئة والجودة

| الأمر | الغرض |
|---|---|
| `just sync` | تثبيت بيئة التشغيل والتطوير المقفلة. |
| `just check` | تشغيل Ruff والتحقق من التنسيق وmypy وجميع الاختبارات. |
| `just fmt` | تطبيق إصلاحات Ruff الآمنة وتنسيق ملفات Python. |
| `just test [pytest args]` | تشغيل جميع الاختبارات أو تمرير مسار/خيارات إلى pytest. |
| `just build` | بناء حزمة Python. |
| `just hooks-install` | تثبيت خطّاف pre-commit للمستودع. |
| `just hooks` | تشغيل جميع فحوص pre-commit فورًا. |

### البحث

| الأمر | الغرض |
|---|---|
| `just report` | كتابة التقرير الرئيسي للإيداع الحالي في `experiments/reports/`. |
| `just causal-run` | تشغيل مجموعة المحفِّز/الضابط المزدوجة (30 فرعًا من نقاط حفظ متوازنة مشتركة) وإعادة بناء الأدلة. |
| `just causal-analyze` | إعادة حساب إحصاءات الخرج والإحصاءات السببية الإقليمية دون محاكاة. |
| `just causal-parts` | قياس الاستجابة مع تعطيل كل مكوّن من مكوّنات الحقل ومع فصل منطقة الخرج. |
| `just causal-ablations` | تشغيل أربعة متغيرات توازن على بذور التطوير ومقارنة استجابات الخرج. |
| `just causal-ablations-report` | إعادة بناء مقارنة المتغيرات من النتائج الجاهزة. |

### الحقل والحالة والأداء

| الأمر | الغرض |
|---|---|
| `just preview` | تشغيل محاكاة قصيرة وكتابة `outputs/field-preview.png`. |
| `just checkpoint [steps] [output]` | حفظ حالة الحقل؛ افتراضيًا بعد 10,000 خطوة. |
| `just resume <checkpoint> <steps> <output>` | متابعة محاكاة محفوظة إلى نقطة حفظ جديدة. |
| `just benchmark` | قياس سرعة خطوات الحقل بالأحجام 64 و128 و200. |

### Sound Protocol

| الأمر | الغرض |
|---|---|
| `just protocol-record [config] [steps] [output]` | تسجيل تدفق JSONL صارم؛ افتراضيًا `field_only.yaml` و128 خطوة. |
| `just protocol-replay [input] [output]` | إعادة تشغيل JSONL دون الحقل وإنتاج ملف WAV تشخيصي. |

### التفاعل الصوتي

| الأمر | الغرض |
|---|---|
| `just audio-devices` | عرض أجهزة الإدخال والإخراج. |
| `just live` | بدء تفاعل بالأدوار عبر الميكروفون وتسجيل الجلسة. |
| `just conversation <input.wav> [output.wav]` | تمرير ملف WAV مسجَّل عبر الحقل وحفظ استجابته. |

## الترخيص

[MIT](LICENSE)

</div>

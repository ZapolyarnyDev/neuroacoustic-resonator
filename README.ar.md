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
السابقة (<span dir="ltr">`0.542`</span> cross-seed balanced accuracy، الوسم <span dir="ltr">`v0-baseline`</span>) نتجت عن تغذية
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

كل وصفة تستدعي <span dir="ltr">`uv run neuroacoustic <command>`</span>؛ للقائمة الكاملة شغّل
<span dir="ltr">`uv run neuroacoustic --help`</span>.

تُكتب المخرجات المولَّدة في <span dir="ltr">`experiments/`</span> و<span dir="ltr">`outputs/`</span> ولا تُضاف إلى المستودع افتراضيًا.

## الأوامر

شغّل <span dir="ltr">`just`</span> لعرض الوصفات العامة الحالية.

### البيئة والجودة

| الأمر | الغرض |
|---|---|
| <span dir="ltr">`just sync`</span> | تثبيت بيئة التشغيل والتطوير المقفلة. |
| <span dir="ltr">`just check`</span> | تشغيل Ruff والتحقق من التنسيق وmypy وجميع الاختبارات. |
| <span dir="ltr">`just fmt`</span> | تطبيق إصلاحات Ruff الآمنة وتنسيق ملفات Python. |
| <span dir="ltr">`just test [pytest args]`</span> | تشغيل جميع الاختبارات أو تمرير مسار/خيارات إلى pytest. |
| <span dir="ltr">`just build`</span> | بناء حزمة Python. |
| <span dir="ltr">`just hooks-install`</span> | تثبيت خطّاف pre-commit للمستودع. |
| <span dir="ltr">`just hooks`</span> | تشغيل جميع فحوص pre-commit فورًا. |

### البحث

| الأمر | الغرض |
|---|---|
| <span dir="ltr">`just report`</span> | كتابة التقرير الرئيسي للإيداع الحالي في <span dir="ltr">`experiments/reports/`</span>. |
| <span dir="ltr">`just causal-run`</span> | تشغيل مجموعة المحفِّز/الضابط المزدوجة (30 فرعًا من نقاط حفظ متوازنة مشتركة) وإعادة بناء الأدلة. |
| <span dir="ltr">`just causal-analyze`</span> | إعادة حساب إحصاءات الخرج والإحصاءات السببية الإقليمية دون محاكاة. |
| <span dir="ltr">`just causal-parts`</span> | قياس الاستجابة مع تعطيل كل مكوّن من مكوّنات الحقل ومع فصل منطقة الخرج. |
| <span dir="ltr">`just causal-ablations`</span> | تشغيل أربعة متغيرات توازن على بذور التطوير ومقارنة استجابات الخرج. |
| <span dir="ltr">`just causal-ablations-report`</span> | إعادة بناء مقارنة المتغيرات من النتائج الجاهزة. |

### الحقل والحالة والأداء

| الأمر | الغرض |
|---|---|
| <span dir="ltr">`just preview`</span> | تشغيل محاكاة قصيرة وكتابة <span dir="ltr">`outputs/field-preview.png`</span>. |
| <span dir="ltr">`just checkpoint [steps] [output]`</span> | حفظ حالة الحقل؛ افتراضيًا بعد 10,000 خطوة. |
| <span dir="ltr">`just resume <checkpoint> <steps> <output>`</span> | متابعة محاكاة محفوظة إلى نقطة حفظ جديدة. |
| <span dir="ltr">`just benchmark`</span> | قياس سرعة خطوات الحقل بالأحجام 64 و128 و200. |

### Sound Protocol

| الأمر | الغرض |
|---|---|
| <span dir="ltr">`just protocol-record [config] [steps] [output]`</span> | تسجيل تدفق JSONL صارم؛ افتراضيًا <span dir="ltr">`field_only.yaml`</span> و128 خطوة. |
| <span dir="ltr">`just protocol-replay [input] [output]`</span> | إعادة تشغيل JSONL دون الحقل وإنتاج ملف WAV تشخيصي. |

### التفاعل الصوتي

| الأمر | الغرض |
|---|---|
| <span dir="ltr">`just audio-devices`</span> | عرض أجهزة الإدخال والإخراج. |
| <span dir="ltr">`just live`</span> | بدء تفاعل بالأدوار عبر الميكروفون وتسجيل الجلسة. |
| <span dir="ltr">`just conversation <input.wav> [output.wav]`</span> | تمرير ملف WAV مسجَّل عبر الحقل وحفظ استجابته. |

## الترخيص

[MIT](LICENSE)

</div>

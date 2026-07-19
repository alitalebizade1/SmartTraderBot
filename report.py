"""
report.py — Human-readable analysis report for SmartTraderBot.

Turns Signal (+ attached Pattern) objects into a plain-language
description of: the trend, the resistance/rising-bottom structure,
and the trade plan (entry/SL/TP1/TP2/RR/confidence).

No trading logic — pure text formatting from already-computed objects.
"""
from __future__ import annotations

from typing import List

from models import Signal


def _fmt_time(t) -> str:
    try:
        return t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(t)


def print_report(signals: List[Signal], swings_count: int, candles_count: int) -> None:
    print("\n" + "=" * 72)
    print(" گزارش تحلیل — SmartTraderBot (الگوی ۳ برخورد به مقاومت / کف صعودی)")
    print("=" * 72)
    print(f" تعداد کندل تحلیل‌شده : {candles_count}")
    print(f" تعداد Swing شناسایی‌شده : {swings_count}")
    print(f" تعداد سیگنال صادرشده : {len(signals)}")

    if not signals:
        print("\n هیچ الگوی معتبری با معیارهای فعلی پیدا نشد.")
        print(" (برای بررسی جزئیات رد شدن‌ها، اسکریپت debug_run.py را با گزینه debug=True اجرا کنید.)")
        print("=" * 72)
        return

    for i, sig in enumerate(signals, 1):
        pat = sig.pattern
        print(f"\n{'─' * 72}")
        print(f" سیگنال شماره {i}")
        print("─" * 72)
        print(f"  زمان سیگنال          : {_fmt_time(sig.candle_time)}")

        if pat is not None:
            if pat.channel_start_price is not None:
                print(f"  شروع کانال (کف اولیه صعود) : {pat.channel_start_price:.2f}")
            if pat.touch1:
                print(f"  برخورد ۱ به مقاومت   : {pat.touch1.price:.2f}  (بار #{pat.touch1.index})")
            if pat.hl1:
                print(f"  کف صعودی ۱ (HL1)     : {pat.hl1.price:.2f}  (بار #{pat.hl1.index})")
            if pat.touch2:
                print(f"  برخورد ۲ به مقاومت   : {pat.touch2.price:.2f}  (بار #{pat.touch2.index})")
            if pat.hl2:
                print(f"  کف صعودی ۲ (HL2)     : {pat.hl2.price:.2f}  (بار #{pat.hl2.index})  ← محل دقیق SL")
            if pat.touch3:
                print(f"  برخورد ۳ به مقاومت   : {pat.touch3.price:.2f}  (بار #{pat.touch3.index})")
            if pat.resistance:
                print(f"  سطح مقاومت (میانگین ۳ برخورد) : {pat.resistance.center:.2f}")

        print()
        print(f"  ورود (Entry)         : {sig.entry:.2f}")
        print(f"  حد ضرر (Stop Loss)   : {sig.sl:.2f}   ← دقیقاً روی HL2")
        print(f"  هدف اول (TP1)        : {sig.tp1:.2f}   ← سطح مقاومت")
        print(f"  هدف دوم (TP2)        : {sig.tp2:.2f}   ← Measured Move (ارتفاع کانال از نقطه ورود)")
        print(f"  ریسک                 : {sig.risk:.2f}")
        print(f"  بازده تا TP2          : {sig.reward:.2f}")
        print(f"  نسبت R:R             : 1 : {sig.risk_reward:.2f}")
        print(f"  اطمینان الگو (Confidence) : {sig.confidence:.1%}")

        print()
        print("  تحلیل روند:")
        print(
            "    یک روند صعودی قوی شناسایی شد که به یک سقف مقاومتی رسید. قیمت سه بار به"
        )
        print(
            "    همان سطح مقاومت برخورد کرد و هر بار پس از اصلاح، کف بالاتری (کف صعودی)"
        )
        print(
            "    تشکیل داد — نشانه‌ی افزایش فشار خرید و تجمع نقدینگی زیر مقاومت."
        )
        print(
            "    سیگنال خرید درست پیش از تلاش سوم برای شکست مقاومت صادر شده، با فرض این"
        )
        print(
            "    که برخورد سوم منجر به شکست و ادامه‌ی روند صعودی خواهد شد."
        )

    print(f"\n{'=' * 72}")
    print(f" جمع کل: {len(signals)} سیگنال معتبر")
    print("=" * 72)

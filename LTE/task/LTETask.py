from crewai import Task
import json
from shabak.LTE.agent.LTEAgent import  greeting_responder_agent, help_internet_agent, \
    context_switch_agent, request_collect_user_info_agent, LTE_Analyst, LTE_detect, detect_buy_service_agent, \
    unknown_agent, detect_user_info_agent, user_info_display_agent, detect_select_account_agent, ask_witch_user_agent,collect_user_info_agent
from shabak.LTE.general.tools import OUTPUT_HTML
from datetime import datetime

INTENTS = [
    "support",
    "buy_service",
    "unknown",
    "greeting",
]
SUBINTENTS = [
    "help_internet",
    "detect_user_info",
    "request_collect_user_info",
    "collect_user_info",
    "ask_which_account",
    "detect_select_account",
    "check_modem",
    "turn_on_modem",
    "setting_modem",
    "edit_setting_modem",
    "return_operator",
]


def create_greeting_response_task(user_message, user_info=None):
    if user_info:
        description = (
            f"پیام کاربر: {user_message}\n"
            f"اطلاعات کاربر: {user_info}\n"
            "یک پاسخ گرم و صمیمی به فارسی بده که:\n"
            "1. جواب احوال‌پرسی کاربر را با لحن دوستانه بده.\n"
            "2. از نام کاربر در پیام استفاده کن تا صمیمی‌تر شود.\n"
            "3. برند شبکیه را کوتاه و غیرمستقیم یادآوری کن (مثلاً در قالب خوشحال بودن از حضورش).\n"
            "4. قابلیت‌ها را ذکر نکن، فقط تاکید کن آماده‌ای هر کمکی لازم داشت انجام بدهی.\n"
            f"{OUTPUT_HTML}"
        )
    else:
        description = (
            f"پیام کاربر: {user_message}\n"
            "یک پاسخ گرم و صمیمی به فارسی بده که:\n"
            "1. جواب احوال‌پرسی کاربر را با لحن دوستانه بده.\n"
            "2. برند شبکیه را به شکلی مثبت و ماندگار معرفی کن.\n"
            "3. قابلیت‌های اصلی شبکیه را ذکر کن (پشتیبانی اینترنت، خرید و ارتقاء سرویس، گزارش خرابی).\n"
            "4. حس اعتماد و اطمینان را منتقل کن.\n"
            "5. در پایان به کاربر بگو که آماده‌ای هر کمکی که لازم دارد را انجام بدهی.\n"
            f"{OUTPUT_HTML}"
        )

    return Task(
        description=description,
        agent=greeting_responder_agent,
        expected_output="متن HTML خام"
    )


def help_internet_task(user_message):
    return Task(
        description=(
            f"پرسش کاربر: {user_message}"
            "یک پاسخ دوستانه و راهنمایی دقیق در زمینه اینترنت "
            "یا اگر سوال خارج از موضوع بود، بگوید که فقط در اینترنت راهنمایی می‌کنید."
            f"{OUTPUT_HTML}"
        )
        ,
        agent=help_internet_agent,
        expected_output="متن HTML خام"
    )


def create_context_switch_task(user_message: str, state: str, last_ai_message: str, ) -> Task:
    return Task(
        description=(
            "شما باید تصمیم بگیرید که پیام جدید کاربر با توجه به لیست intent های قبلی و متن پیام فعلی در کدام مسیر قرار می‌گیرد.\n"
            f"json مشکلات کاربر را بخوان: {state.get('lte_status')}\n"
            f"لیست مسیرهای ممکن: {', '.join(INTENTS)}\n\n"
            f"لیست intentهای قبلی: {state.get('history_intent', [])}\n"
            f"مسیر فعلی (next_node): {state.get('next_node') if state.get('next_node') else 'ندارد'}\n"
            f"آخرین پیام هوش مصنوعی (اگر موجود باشد): {last_ai_message if last_ai_message else 'ندارد'}\n"
            f"پیام جدید کاربر: {user_message}\n\n"
            "قوانین تشخیص:\n"
            "- اگر پیام کاربر ادامه intent قبلی است، خروجی همان intent می‌باشد.\n"
            " گام 1:مقادیر کلیدی موجود در json مانند network_status، modem_status, signal_issue , sim_issue را تحلیل کن."
            "گام 2: بر اساس مقدارها تشخیص بده مشکل از کدام بخش است"
            "حتما ترتیب زیر را رعایت کن:"
            " 1- اگر network_status = 'offline' بود کاربر در حالت آفلاین قرار داد و باید مودم بررسی شود و خروجی check_modem است."
            "2- اگر modem_status='off' بود باید مودم روشن شود و خروجی  turn_on_modem است."
            # "3- اگرmodem_on_know='no' یعنی کاربر از چگونگی روشن کردن مودم اطلاعی ندارد و خروجی education_turn_on_modem است."
            "4- اگر modem_status='on' بود باید تنظیمات مودم مورد بررسی قرار گیرد و خروجی  setting_modem است."
            # "5- اگر modem_config_know='no' یعنی کاربر اطلاع از چگونگی دسترسی به تنظیمات مودم ندارد و خروجی education_setting_modem است."
            "6- اگر modem_config='error' یعنی تنظیمات مودم مشکل دارد و خروجی edit_setting_modem است."
            "7- اگر modem_config='ok' یعنی تنظیمات مودم بدون مشکل است و خروجی return_operator است."
            " 8- در غیر این صورت، خروجی 'unknown' است."
            "⚠️ خروجی باید فقط نام یکی از نودها باشد، بدون هیچ متن اضافی."
        ),
        agent=context_switch_agent,
        expected_output="یکی از نام‌های لیست نودها"
    )


def create_support_switch_task(user_message: str, state: str, last_ai_message: str) -> Task:
    return Task(
        description=(
            "شما باید تصمیم بگیرید که پیام جدید کاربر با توجه به لیست sub intents های قبلی و متن پیام فعلی در کدام مسیر قرار می‌گیرد.\n"
            f"لیست مسیرهای ممکن: {', '.join(SUBINTENTS)}\n\n"
            f"لیست intent های قبلی: {state.get('history_intent', [])}\n"
            f"مسیر فعلی (next_node): {state.get('next_node') if state.get('next_node') else 'ندارد'}\n"
            f"آخرین پیام هوش مصنوعی (اگر موجود باشد): {last_ai_message if last_ai_message else 'ندارد'}\n"
            f"پیام جدید کاربر: {user_message}\n\n"
            "قوانین تشخیص:\n"
            "- اگر پیام کاربر ادامه intent قبلی است، خروجی همان intent می‌باشد.\n"
            " گام 1:مقادیر کلیدی موجود در json مانند network_status، modem_status, signal_issue , sim_issue را تحلیل کن."
            "گام 2: بر اساس مقدارها تشخیص بده مشکل از کدام بخش است"
            "حتما ترتیب زیر را رعایت کن:"
            " 1- اگر network_status = 'offline' بود کاربر در حالت آفلاین قرار داد و باید مودم بررسی شود و خروجی check_modem است."
            "2- اگر modem_status='off' بود باید مودم روشن شود و خروجی  turn_on_modem است."
            # "3- اگرmodem_on_know='no' یعنی کاربر از چگونگی روشن کردن مودم اطلاعی ندارد و خروجی education_turn_on_modem است."
            "4- اگر modem_status='on' بود باید تنظیمات مودم مورد بررسی قرار گیرد و خروجی  setting_modem است."
            # "5- اگر modem_config_know='no' یعنی کاربر اطلاع از چگونگی دسترسی به تنظیمات مودم ندارد و خروجی education_setting_modem است."
            "6- اگر modem_config='error' یعنی تنظیمات مودم مشکل دارد و خروجی edit_setting_modem است."
            "7- اگر modem_config='ok' یعنی تنظیمات مودم بدون مشکل است و خروجی return_operator است."
            " 8- در غیر این صورت، خروجی 'unknown' است."
            "⚠️ خروجی باید فقط نام یکی از نودها باشد، بدون هیچ متن اضافی."
        ),
        agent=context_switch_agent,
        expected_output="یکی از نام‌های لیست نودها"
    )


def request_collect_user_info_task(user_message: str) -> Task:
    return Task(
        description=(
            f"پیام کاربر: {user_message}\n\n"
            "تو یک پشتیبان اینترنت هستی. "
            "پاسخ باید فقط یک جمله مودبانه باشد که توضیح دهد برای بررسی و رسیدگی به مشکل کاربر نیاز به کد ده‌رقمی (شماره تلفن) داریم. "
            "⚠️ ممنوع است: سلام کردن، احوال‌پرسی، معرفی، شوخی، ایموجی، یا جملات کلی مثل «در خدمت شما هستیم». "
            "پاسخ باید کوتاه، مستقیم، و مرتبط با پیام کاربر باشد. "
            "فقط از کاربر بخواه کد ده‌رقمی را بدهد.\n"
            f"{OUTPUT_HTML}"
        ),
        agent=request_collect_user_info_agent,
        expected_output="متن HTML خام"
    )


def collect_user_info_task(user_input: str) -> Task:
    return Task(
        description=f"""
    کاربر این اطلاعات را داده:
    {user_input}

    قوانین:
    1- فقط شماره تلفن 10 رقمی (username) را واکشی کن.
    2- اگر شماره با صفر شروع شد، آن صفر را حذف کن.
    3- اگر هیچ شماره‌ای در متن وجود نداشت → مقدار را null برگردان.
    4- خروجی فقط و فقط یک JSON معتبر با کلید "username" باشد.

    مثال:
    - ورودی: "شماره من 09121234567 است"
      خروجی: {{"username": "9121234567"}}

    - ورودی: "میخوام اطلاعات اکانتم رو ببینم"
      خروجی: {{"username": null}}
    """,
        agent=collect_user_info_agent,
        expected_output="یک JSON معتبر با کلید username"
    )


def create_display_user_info_task(user_info: dict) -> Task:
    """
    user_info: دیکشنری اطلاعات کاربر
    خروجی: متن HTML مرتب و دوستانه
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = (
        f"اطلاعات کاربر به صورت دیکشنری داده شده:\n{user_info}\n\n"

        f"تاریخ و زمان فعلی سیستم: {now}\n\n"
        "این اطلاعات را به صورت HTML تمیز، مرتب و دوستانه نمایش بده. "
        "⚠️ سلام یا جملات کلیشه‌ای آغازین ننویس. مستقیم فقط به موضوع کاربر بپرداز.\n\n"

        "🔹 منطق نمایش وضعیت:\n"
        "- مقدار فیلد Pl_Status وضعیت سرویس کاربر است.\n"
        "- اگر «بهره برداری» بود:\n"
        "   1. تاریخ انقضا (PL_ActiveTo) را بررسی کن. "
        "      اگر از تاریخ فعلی گذشته باشد، اعلام کن که سرویس تمام شده. "
        "      اگر هنوز نگذشته، تاریخ اعتبار را نمایش بده.\n"
        "   2. حجم (PL_Hajm) و توضیحات سرویس را بررسی کن:\n"
        "      - اگر در توضیحات سرویس (Crm_FinalSefaresh یا PL_FinalSefareshDescription) کلمه «غیرحجمی» بود → حجم نامحدود است.\n"
        "      - در غیر این صورت:\n"
        "          - اگر PL_Hajm > 0 → مقدار باقی‌مانده را نمایش بده.\n"
        "          - اگر PL_Hajm = 0 یا کمتر → اعلام کن که حجم تمام شده است.\n"
        "- اگر Pl_Status مقدار دیگری بود (مثل قطع، تعلیق، لغو): "
        "   همان وضعیت را نمایش بده و توضیح بده سرویس در حال حاضر فعال نیست.\n\n"

        "🔹 نمایش تاریخ‌ها:\n"
        "- تمام تاریخ‌ها (مثل PL_ActiveTo یا RanjeDate) را به تقویم شمسی نمایش بده.\n"
        "- فرمت تاریخ: روز/ماه/سال - ساعت:دقیقه (مثال: 22/07/1404 - 13:10).\n\n"
        f"{OUTPUT_HTML}"
    )

    return Task(
        description=description,
        agent=user_info_display_agent,
        expected_output="متن HTML خام"
    )


def create_detect_user_info_task(user_message: str, user_accounts: list) -> Task:
    accounts_str = "\n".join(
        [f"- {a.get('username')}" for a in user_accounts]) if user_accounts else "❌ هیچ اکانتی وجود ندارد."

    return Task(
        description=(
            f"لیست اکانت‌های موجود کاربر:\n{accounts_str}\n\n"
            f"پیام کاربر:\n{user_message}\n\n"
            "دستورالعمل:\n"
            "1. ابتدا بررسی کن آیا کاربر درباره‌ی اکانت صحبت کرده (پشتیبانی، تغییر، مشکل، ورود، خروج، بررسی و ...).\n"
            "   اگر هیچ اشاره‌ای به اکانت نکرده و فقط گفت‌وگو ادامه دارد → خروجی 'keep_current'.\n\n"
            "2. در غیر این صورت (هرگونه اشاره به اکانت یا نیاز به تغییر):\n"
            "   - اگر عدد ۱۰ رقمی (فارسی یا انگلیسی) در پیام وجود دارد:\n"
            "       → آن را به عنوان username بگیر (نرمالایز کن، صفر اول را حذف کن)\n"
            "         • اگر در لیست اکانت‌ها هست → action='select'\n"
            "         • اگر نیست → action='new_username'\n\n"
            "   - اگر عدد ۱۰ رقمی وجود ندارد:\n"
            "       • اگر لیست اکانت‌ها خالی است → action='new_username'\n"
            "       • اگر لیست وجود دارد → action='ask_select' و message شامل HTML لیست باشد\n\n"
            "⚠️ نکته بسیار مهم:\n"
            "اگر لیست اکانت‌ها خالی است، حتی اگر هیچ اشاره مستقیمی به اکانت در پیام نباشد، نمی‌توان 'keep_current' داد.\n"
            "در این حالت همیشه باید خروجی 'switch_account' با action='new_username' تولید شود.\n\n"
            # "اگر کاربر اطلاعات درستی که شامل username باشد یا عدد ده رقمی وارد نکرد برای تکمیل اطلاعات حتما از او بخواهید نام کاربری اش را وارد کند و در username قرارش بده و action='new_username'."
            "خروجی فقط یک JSON معتبر با کلیدهای intent, action, username, message باشد."
        ),
        agent=detect_user_info_agent,
        expected_output="یک JSON معتبر با کلیدهای intent, action, username, message"
    )


def ask_witch_user_task(accounts_str: str) -> Task:
    return Task(
        description=f"از کاربر بخواه که یکی از اکانت‌های زیر را انتخاب کند:\n{accounts_str}\n"
                    f"به کاربر بگو فقط شماره انتخاب را وارد کند."
                    f"{OUTPUT_HTML}",
        agent=ask_witch_user_agent,
        expected_output="متن HTML خام"
    )


def detect_select_account_task(user_input: str, accounts_str: str) -> Task:
    return Task(
        description=f"کاربر این پیام را فرستاده: '{user_input}'\n"
                    f"لیست اکانت‌ها:\n" +
                    accounts_str +
                    "-خروجی میتونه ایندکس یکی از اکانت ها باشه"
                    "-خروجی میتونه به صورت متنی مشخص کنه مثلا:اولی،دومی"
                    "-خروجی میتونه یه قسمت از شماره یا کامل باشه",
        agent=detect_select_account_agent,
        expected_output="یک عدد که نشان‌دهنده شماره اکانت انتخاب شده توسط کاربر است."
    )


def unknown_task(user_message: str) -> Task:
    return Task(
        description=(
            f"پیام کاربر: {user_message}\n\n"
            "تو یک پشتیبان اینترنت هستی. این پیام در هیچ مسیر مشخصی قرار نمی‌گیرد.\n"
            "وظیفه تو این است که:\n"
            "- مودبانه و دوستانه به کاربر بگی که پیامش واضح نیست.\n"
            "- بر اساس متن پیام کاربر، اگر سرنخی وجود داشت (مثلاً کلمه‌ای درباره اینترنت، خرید، قطعی و ...)، در پاسخ اشاره کنی.\n"
            "- در نهایت از او بخواهی واضح‌تر توضیح بدهد یا انتخاب کند (پشتیبانی سرویس فعلی یا خرید/ارتقاء سرویس جدید).\n"
            "- پاسخ باید کوتاه، طبیعی، و شبیه چت انسانی باشد (مثل ChatGPT).\n"
            "- ⚠️ خروجی فقط باید در قالب HTML باشد (بدون متن اضافی).\n"
            f"{OUTPUT_HTML}"
        ),
        agent=unknown_agent,
        expected_output="متن HTML خام"
    )


def detect_buy_service_task(
        user_message: str,
        previous_filters_json: str,  # رشته‌ی JSON همین اسکیما؛ اگر نداشتی "null" بده
        bounds: dict,  # {"speed": {"min":..,"max":..}, "volume":..., "duration_days":..., "budget":...}
        units: dict = None  # {"currency":"TOMAN"} مثلا
) -> Task:
    bounds_str = json.dumps(bounds, ensure_ascii=False)
    print("bounds >>>", bounds_str)
    prev_str = previous_filters_json if previous_filters_json else "null"
    print("prev>>>>", prev_str)
    units_str = json.dumps(units or {"currency": "TOMAN"}, ensure_ascii=False)

    print("task str", str)
    description = (
        f"""
        [هدف]
        - تحلیل پیام کاربر برای استخراج فیلترهای انتخاب سرویس اینترنت.
        -اگر جمله کاربر به صورت کلی بود سعی کن نیتش رو تشخیص بدی اگر به صورت کلی حرف نزده باشه نیازی نیست نیت رو تشخیص بدی
        مثال‌ها "video", "download", "browsing", "gaming", "remote_work", "public".
        -اگر کاربر گفته باشه یک سرویس معمولی یا عمومی باید usage_type بشه public در غیر اینصورت اگر کاربر هیچ اشاره ای به نوع سرویس نکند نیازی نیست usage_type پر شود
        - اگر نیت تشخیص داده شد usage_type، یک سری فیلتر پیش‌فرض بساز (سرعت، حجم، مدت، بودجه) طبق جدول زیر:

        پیش‌فرض‌ها:
        video → speed min=8, volume min=200
        download → speed min=8, volume min=500
        browsing → speed 2–8, volume 50–150
        gaming → speed min=8, volume min=100
        remote_work → speed min=8, volume min=150
        public → speed min=4, volume min=100
        - بودجه باید همیشه عددی باشد (min/max به تومان بر اساس کران منطقی).
        - اگر کاربر عباراتی مثل "ارزون"، "کم‌هزینه"، "پایین" گفت → مقدار بودجه را در نزدیکی حداقل قرار بده (مثلا min=null, max=حد میانی پایین).
- اگر گفت "متوسط"، "قابل قبول" → بودجه را در بازه‌ی میانه تنظیم کن (مثلا min=null, max=میانه bounds).        
- اگر گفت "گرون"، "بالا"، "مهم نیست" → بودجه را نزدیک به حداکثر یا کل بازه قرار بده.        
- اگر صراحتاً عدد گفت (مثلا "تا ۲ میلیون") → همان عدد را اعمال کن (clip داخل کران منطقی).        
        - اگر کاربر صریحاً چیزی گفت (مثل "ارزون" → budget پایین، "سه ماهه" → duration_days=90، "بالای 200 گیگ" → volume min=200)،
          این مقدار باید جایگزین پیش‌فرض شود (override).
        - اگر چیزی نگفت، از پیش‌فرض‌ها استفاده کن.
        - اگر گفت "مهم نیست/فرقی نداره"، status="ignored".
        - خروجی باید با JSON قبلی merge شود.
        -اگر به صورت مستقیم چیزی رو تغییر نداد داده های قبلی رو نگه دار
        - -اگر کاربر بقیه موارد براش مهم نبود یا گفت بقیه تنظیمات رو نمیخوام باید بقیه موارد رو unset کنی
        -اگر کاربر نخواست که جزییات بیشتری بده یا اینکه تایید کرد داده های قبلی را باید فیلد complete_query را true ست کنی در غیر اینصورت false کن 
        [ورودی‌ها]
        - پیام کاربر:
        {user_message}

        - JSON قبلی (اگر null باشد یعنی اولیه):
        {prev_str}

        - کران‌های منطقی:
        {bounds_str}

        - واحدها:
        {units_str}

        [اسکیما خروجی — دقیقاً همین]
        {{
          "filters": {{
            "speed":        {{ "min": null, "max": null, "status": "unset" }},
            "volume":       {{ "min": null, "max": null, "status": "unset" }},
            "duration_days":{{ "min": null, "max": null, "status": "unset" }},
            "budget":       {{ "min": null, "max": null, "status": "unset" }}
          }},
          "usage_type":       {{ "value": null, "status": "unset" }},
          "special_features": {{ "night_free": null, "with_modem": null, "other": [], "status": "unset" }},
          "user_intent":      {{ "complete_query": false, "explanation_request": false, "help_more": false }}
        }}

        [الزام‌ها]
        - خروجی فقط یک JSON معتبر باشد، بدون متن اضافه.
        - حتماً از پیش‌فرض usage_type شروع کن و فقط در صورت وجود داده صریح از کاربر override کن.
        - merge با JSON قبلی را انجام بده.
        """
    )
    return Task(
        description=description,
        agent=detect_buy_service_agent,  # یک agent سخت‌گیر با temperature=0
        expected_output="یک JSON معتبر مطابق اسکیما"
    )


# lte task
def create_context_switch_task_lte(user_message: str, state: str, last_ai_message: str) -> Task:
    return Task(
        description=(
            "شما باید تصمیم بگیرید که پیام جدید کاربر با توجه به لیست intentهای قبلی و متن پیام فعلی در کدام مسیر قرار می‌گیرد.\n"
            f"لیست مسیرهای ممکن: {', '.join(INTENTS)}\n\n"
            f"لیست intentهای قبلی: {state.get('history_intent', [])}\n"
            f"مسیر فعلی (next_node): {state.get('next_node') if state.get('next_node') else 'ندارد'}\n"
            f"آخرین پیام هوش مصنوعی (اگر موجود باشد): {last_ai_message if last_ai_message else 'ندارد'}\n"
            f"پیام جدید کاربر: {user_message}\n\n"
            "قوانین تشخیص:\n"
            "اگر کاربر اطلاعاتی درباره مودم نداد باید از او بپرسی مودم روشن است یا خیر"
            "اگر پیام آموزشی است (مودم، تنظیمات، کانفیگ) به او بطور کامل توضیح بده."
            "اگر موردم روشن بود باید تنظیمات مودم بررسی شود"
            "اگر تنظیمات مودم مشکلی نداشت احتما وجود مشکل در آنتن دهی یا سیم کارت است که باید به کاربر اطلاع دهید مشکل او به اپراتور ارجاع داده شده و منتظر پاسخ بماند."
            "⚠️ خروجی باید مودبانه حرفه ای و دقیق باشد."
        ),
        agent=LTE_Analyst,
        expected_output="متن HTML خام"
    )


def LTE_detect_task(user_message: str, ) -> Task:
    return Task(
        name="LTE_Detection_Task",
        description=(
            f"این تسک وظیفه دارد ورودی کاربر {user_message} را به‌صورت دقیق تحلیل کند "
            "تا نوع مشکل در شبکه LTE شناسایی شود. "
            "هدف، تولید یک خروجی JSON معتبر مطابق با اسکیمای تعیین‌شده است تا سیستم بتواند بر اساس آن تصمیم‌گیری خودکار انجام دهد.\n\n"
            "📋 مراحل اجرای تسک:\n"
            "1️⃣ متن پیام کاربر را بخوان و تشخیص بده آیا او به‌طور ضمنی یا صریح خودش را معرفی کرده است (با IP یا نام کاربری).\n"
            "2️⃣ تعیین کن آیا اتصال شبکه LTE برقرار است یا خیر (بر اساس کلیدواژه‌هایی مثل 'اینترنت ندارم' → offline یا 'اینترنت دارم' → online).\n"
            "3️⃣ وضعیت مودم را بررسی کن (روشن، خاموش یا نامشخص).\n"
            "4️⃣ بسنج آیا کاربر از چگونگی روشن کردن یا تنظیم مودم آگاهی دارد (عباراتی مثل 'نمی‌دونم مودم چجوری کار می‌کنه' → no).\n"
            "5️⃣ تشخیص بده آیا تنظیمات مودم درست است یا خیر (عباراتی مثل 'تنظیمات مودم به هم ریخته' → error).\n"
            "6️⃣ وضعیت سیم‌کارت و آنتن را از متن استنباط کن (درصورت اشاره به 'سیم‌کارت' یا 'آنتن').\n"
            "7️⃣ اگر اطلاعات کافی برای هر بخش وجود ندارد، مقدار آن را 'unknown' قرار بده.\n\n"
            "⚙️ الزامات خروجی:\n"
            "- خروجی فقط باید JSON باشد (بدون هیچ متن اضافی).\n"
            "- تمام فیلدهای زیر باید در JSON وجود داشته باشند:\n"

            "{\n"
            '  "user_identified": "<yes | no | unknown>",\n'
            '  "network_status": "<online | offline | unknown>",\n'
            '  "modem_status": "<on | off | unknown>",\n'
            '  "modem_on_know": "<yes | no | unknown>",\n'
            '  "modem_config_know": "<yes | no | unknown>",\n'
            '  "modem_edit_config_know": "<yes | no | unknown>",\n'
            '  "modem_config": "<ok | error | unknown>",\n'
            '  "sim_issue": "<ok | error | unknown>",\n'
            '  "signal_issue": "<ok | error | unknown>"\n'
            "}\n\n"
            "🎯 هدف نهایی: تولید یک خلاصه ساختاریافته از وضعیت کاربر در شبکه LTE که سیستم بتواند از آن برای عیب‌یابی خودکار استفاده کند."
        ),
        agent=LTE_detect,
        expected_output=(
            "JSON معتبر و کامل مطابق با اسکیمای داده‌شده، بدون هیچ توضیح اضافی یا متن غیرساخت‌یافته."
        ),
    )


def check_modem_task(user_message):
    return Task(
        description=(
            f"پرسش کاربر: {user_message}"
            " کاربر را دقیق راهنمایی کن تا بررسی کند مودم lte اش روشن است یا خیر "
            "یا اگر سوال خارج از موضوع بود، بگوید که فقط در اینترنت راهنمایی می‌کنید."
            f"{OUTPUT_HTML}"
        )
        ,
        agent=LTE_Analyst,
        expected_output="متن HTML خام"
    )


def turn_on_modem_task(user_message):
    return Task(
        description=(
            f"پرسش کاربر: {user_message}"
            " به کاربر اطلاع دهید که برای دسترسی به اینترنت باید مودم خود را روشن کند اگر نیاز به راهنمایی داشت او را کامل راهنمایی کنید "
            "یا اگر سوال خارج از موضوع بود، بگوید که فقط در اینترنت راهنمایی می‌کنید."
            f"{OUTPUT_HTML}"
        )
        ,
        agent=LTE_Analyst,
        expected_output="متن HTML خام"
    )


def setting_modem_task(user_message):
    return Task(
        description=(
            f"پرسش کاربر: {user_message}"
            " از کاربر بخواهید تنظیمات مودم LTE اش را برای شما بفرستد "
            " در صورت نیاز به آموزش برای دریافت تنظیمات مودم، به او آموزشی با مثال و کامل و ساده و قابل فهم ارائه دهید"
            "یا اگر سوال خارج از موضوع بود، بگوید که فقط در اینترنت راهنمایی می‌کنید."
            f"{OUTPUT_HTML}"
        )
        ,
        agent=LTE_Analyst,
        expected_output="متن HTML خام"
    )


def edit_setting_modem_task(user_message):
    return Task(
        description=(
            f"پرسش کاربر: {user_message}"
            " تنظیمات صحیح مودم LTE را برای کاربر ارسال کنید و به او بگویید تنظیمات را بر روی مودم خودش اجرا کند."
            "در صورت نیاز آموزش های لازم در زمینه تنظیمات مودم LTE برای او ارسا ل کنید."
            "یا اگر سوال خارج از موضوع بود، بگوید که فقط در اینترنت راهنمایی می‌کنید."
            f"{OUTPUT_HTML}"
        )
        ,
        agent=LTE_Analyst,
        expected_output="متن HTML خام"
    )


def return_operator_task(user_message):
    return Task(
        description=(
            f"پرسش کاربر: {user_message}"
            " به کاربر اطلاع دهید که مشکل ممکن است در زمینه های دیگر مثل اشکال  سیم کارت یا آنتن دهی می باشد و مشکل در حال پیگیری توسط اپراتور های شرکت شبکیه است."
            "یا اگر سوال خارج از موضوع بود، بگوید که فقط در اینترنت راهنمایی می‌کنید."
            f"{OUTPUT_HTML}"
        )
        ,
        agent=LTE_Analyst,
        expected_output="متن HTML خام"
    )

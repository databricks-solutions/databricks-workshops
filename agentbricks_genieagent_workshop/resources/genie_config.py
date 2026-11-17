"""The production configuration of the Kasterwell Clinic Ops Genie Agent: one source for the presenters' space
(kasterwell_health.gold, 8 tables), the kit's "Skip the clicks" space (your schema, 5 tables) and notebook 01's lesson.

    build_space(schema, tables)  -> serialized_space dict (descriptions, synonyms, joins, measures, filters,
                                    example SQL, instructions, benchmarks), limited to the tables you pass
    column_comments(table)       -> {column: description}, applied to Unity Catalog so SQL users see the same text

Every number the demos quote is a benchmark here, with ground-truth SQL.
"""

DATA_START, DATA_END = "2024-11-01", "2026-10-31"
LATEST_WEEK, LATEST_MONTH = "2026-10-26", "2026-10-01"
RECENT_3M = "2026-08-01"  # first month of "the last 3 months of data" (Aug-Oct 2026)

# ── Tables and columns ────────────────────────────────────────────────────────────────────────────────────────────
# column: (description, synonyms, value matching, exclude from Genie)
_ = None
TABLES = {
    "dim_clinic": ("One row per clinic (Kasterwell's customer, also called tenant) with its current subscription. "
                   "Join every other table to it on tenant_id.", {
        "tenant_id": ("Clinic identifier; join key for every table", ["clinic id", "tenant", "customer id"], False, False),
        "clinic_name": ("Clinic display name", ["clinic", "practice", "customer", "account"], True, False),
        "specialty": ("Clinical specialty: primary_care, behavioral_health, urgent_care, pediatrics, womens_health, dermatology",
                      ["service line"], True, False),
        "region": ("Sales region: West, Central or East", ["territory"], True, False),
        "state": ("US state, 2-letter code", [], True, False),
        "city": ("City", [], True, False),
        "go_live_date": ("Date the clinic started using the platform", ["start date", "launch date"], False, False),
        "plan_tier": ("Current plan: Starter, Growth or Enterprise", ["plan", "tier", "package"], True, False),
        "seats": ("Licensed clinician seats on the current contract", ["licenses", "licensed seats"], False, False),
        "addons": ("Comma-separated add-ons turned on: sms_reminders, ai_scribe (empty = none)", ["add-ons"], False, False),
        "mrr_usd": ("Monthly recurring revenue, USD", ["MRR", "monthly revenue"], False, False),
        "arr_usd": ("Annual recurring revenue, USD (12 x MRR); the revenue at stake at renewal", ["ARR", "contract value", "annual revenue"], False, False),
        "renewal_date": ("Next contract renewal date (all are after the data end, 2026-10-31)", ["renewal", "contract end"], False, False),
        "subscription_status": ("active or at_risk (flagged by customer success)", ["status"], True, False),
        "sms_reminders_enabled_date": ("Date the clinic turned on SMS appointment reminders; null = not enabled", ["sms start"], False, False),
        "ai_scribe_enabled_date": ("Date the clinic turned on AI Scribe; null = not enabled", ["scribe start", "AI Scribe adoption date"], False, False),
    }),
    "dim_payer": ("Insurance payers billed by Kasterwell clinics (all fictional). Join to claims_monthly on payer_id.", {
        "payer_id": ("Payer identifier", [], False, False),
        "payer_name": ("Payer name, e.g. Larkspan Health Plan", ["payer", "insurer", "health plan", "insurance"], True, False),
        "payer_type": ("commercial, medicare_advantage, medicaid_managed_care or self_pay", ["line of business"], True, False),
        "regions": ("Regions where the payer operates (ALL = nationwide)", [], True, False),
    }),
    "claims_monthly": ("Monthly insurance claims per clinic, payer, procedure (CPT) code and denial reason. One row per "
                       "month x clinic x payer x CPT code x denial reason. Recent months still have pending claims.", {
        "month": ("Service month (first day of the month)", ["service month", "date"], False, False),
        "tenant_id": ("Clinic identifier", ["clinic id"], False, False),
        "region": ("Clinic's sales region: West, Central or East", [], True, False),
        "specialty": ("Clinic's specialty, e.g. behavioral_health", [], True, False),
        "payer_id": ("Payer identifier", [], False, False),
        "payer_name": ("Payer name, e.g. Larkspan Health Plan", ["payer", "insurer", "health plan"], True, False),
        "payer_type": ("commercial, medicare_advantage or medicaid_managed_care", [], True, False),
        "cpt_code": ("CPT procedure code, e.g. 90834 (psychotherapy, 45 min), 99215 (established visit, high complexity)",
                     ["procedure code", "CPT"], True, False),
        "cpt_description": ("What the CPT code is", ["procedure"], True, False),
        "cpt_category": ("E/M established, E/M new, Preventive, Behavioral, Lab or Procedure", ["code category"], True, False),
        "denial_reason_code": ("CARC denial reason on denied claims, e.g. CO-197 (prior authorization absent), CO-16 "
                               "(missing information), CO-50 (not medically necessary), CO-109 (wrong payer), CO-97 "
                               "(bundled), CO-29 (timely filing); null for paid or pending claims",
                               ["denial reason", "CARC", "reason code"], True, False),
        "claims": ("Claims submitted (paid + denied + pending)", ["claim volume", "submitted claims"], False, False),
        "decided_claims": ("Claims the payer has decided (paid, denied or under appeal); excludes pending", ["adjudicated claims"], False, False),
        "denied_claims": ("Claims denied, including those under appeal", ["denials"], False, False),
        "denial_rate": ("Row-level denial rate; never average it. Use the denial_rate measure", [], False, True),
        "billed_usd": ("Billed amount, USD", ["charges", "billed"], False, False),
        "paid_usd": ("Amount paid by the payer, USD", ["collections", "paid"], False, False),
        "denied_billed_usd": ("Billed value of denied claims, USD: the revenue at risk from denials", ["revenue at risk", "denied dollars"], False, False),
    }),
    "appointment_access": ("Monthly appointment access per clinic by appointment type, visit mode, time of day and "
                           "whether an SMS reminder was sent. Use for no-shows and wait times by segment.", {
        "month": ("Month (first day of the month)", [], False, False),
        "tenant_id": ("Clinic identifier", ["clinic id"], False, False),
        "specialty": ("Clinic's specialty, e.g. behavioral_health", [], True, False),
        "region": ("Clinic's sales region: West, Central or East", [], True, False),
        "appointment_type": ("Appointment type, e.g. Therapy session (45 min), Sick visit", ["visit type"], True, False),
        "visit_mode": ("video or in_person", ["virtual", "telehealth", "modality"], True, False),
        "time_of_day": ("morning (before 12:00), afternoon (12:00-17:00) or evening (17:00 and later)", ["daypart"], True, False),
        "reminder_sent": ("True if an SMS reminder was sent for these appointments", ["SMS reminder", "reminded"], False, False),
        "appointments": ("Booked appointments", ["bookings", "scheduled visits"], False, False),
        "no_shows": ("Appointments the patient missed", ["missed appointments", "DNA"], False, False),
        "no_show_rate": ("Row-level no-show rate; never average it. Use the no_show_rate measure", [], False, True),
        "avg_wait_minutes": ("Average minutes patients waited in this row; weight by appointments (avg_wait measure)", ["wait time"], False, False),
    }),
    "tenant_health_weekly": ("Weekly customer health per clinic: active clinicians vs seats, logins, visits, support "
                             "tickets and CSAT. Weeks start on Monday; the latest week is 2026-10-26. Use for churn and "
                             "renewal risk (playbook CS-02).", {
        "week": ("Week start date (Monday)", [], False, False),
        "tenant_id": ("Clinic identifier", ["clinic id"], False, False),
        "clinic_name": ("Clinic display name", ["clinic"], True, False),
        "seats": ("Licensed clinician seats", [], False, False),
        "weekly_active_clinicians": ("Distinct clinicians active on the platform that week", ["active clinicians"], False, False),
        "seat_utilization": ("weekly_active_clinicians / seats for one clinic-week; can exceed 1 when a clinic uses more "
                             "clinicians than it licenses (an expansion signal)", ["adoption", "seat usage", "utilization"], False, False),
        "logins": ("Platform logins that week", [], False, False),
        "visits": ("Completed visits that week", ["weekly visits"], False, False),
        "tickets": ("Support tickets opened that week", ["support tickets", "cases"], False, False),
        "p1_tickets": ("Priority-1 (outage-level) tickets opened that week", ["P1s", "P1 tickets", "outages", "sev1"], False, False),
        "avg_csat": ("Average CSAT (1-5) of rated tickets that week; null when no ticket was rated", ["CSAT", "satisfaction"], False, False),
    }),
    "clinic_daily_ops": ("Daily operations per clinic: bookings, completed visits, no-shows, cancellations, waits, "
                         "video and AI Scribe visits, active clinicians.", {
        "date": ("Visit date", ["day"], False, False),
        "tenant_id": ("Clinic identifier", ["clinic id"], False, False),
        "appointments_booked": ("All appointments booked for that day", ["bookings"], False, False),
        "visits_completed": ("Visits that happened", ["completed visits", "encounters"], False, False),
        "no_shows": ("Booked appointments the patient missed", ["missed appointments"], False, False),
        "cancellations": ("Booked appointments cancelled", [], False, False),
        "no_show_rate": ("Row-level no-show rate; never average it. Use no_shows / appointments_booked", [], False, True),
        "avg_wait_minutes": ("Average minutes patients waited that day; weight by visits_completed", ["wait time"], False, False),
        "p90_wait_minutes": ("90th-percentile wait that day, minutes; do not average across days for a network figure", ["p90 wait"], False, False),
        "video_visits": ("Completed visits held by video", ["virtual visits", "telehealth visits"], False, False),
        "ai_scribe_visits": ("Completed visits documented with AI Scribe", [], False, False),
        "active_clinicians": ("Clinicians with at least one completed visit that day", [], False, False),
    }),
    "clinician_productivity_monthly": ("Monthly throughput per clinician: visits, days worked, AI Scribe share and "
                                       "note-writing time. Rows with active_days = 0 are months the clinician did not work.", {
        "month": ("Month (first day of the month)", [], False, False),
        "tenant_id": ("Clinic identifier", ["clinic id"], False, False),
        "clinician_id": ("Pseudonymous clinician identifier", ["provider id"], False, False),
        "visits": ("Completed visits that month", [], False, False),
        "active_days": ("Days the clinician completed at least one visit (0 = did not work that month)", ["days worked"], False, False),
        "visits_per_active_day": ("Row-level ratio; use the visits_per_active_day measure", [], False, True),
        "ai_scribe_share": ("Share of the clinician's visits documented with AI Scribe (0-1)", ["scribe usage"], False, False),
        "avg_note_minutes": ("Average minutes to finish a visit note; weight by visits (note_minutes measure)", ["documentation time", "charting time"], False, False),
    }),
    "revenue_monthly": ("Kasterwell's own monthly invoice to each clinic (its SaaS revenue), as of 2026-10-31.", {
        "month": ("Invoice month (first day of the month)", [], False, False),
        "tenant_id": ("Clinic identifier", ["clinic id"], False, False),
        "invoiced_usd": ("Invoice amount, USD", ["invoice amount", "billings"], False, False),
        "invoice_status": ("paid, open (not yet due) or overdue (past due, unpaid)", [], True, False),
        "due_date": ("Invoice due date", [], False, False),
        "paid_date": ("Date paid; null if unpaid as of 2026-10-31", [], False, False),
        "days_late": ("Days paid after the due date (negative = early); null if unpaid", ["days past due"], False, False),
    }),
}


def column_comments(table):
    return {c: d for c, (d, *_x) in TABLES[table][1].items()}


# ── CS-02 renewal risk: the playbook's exact definition ──────────────────────────────────────────────────────────
RENEWAL_DEFINITION = """Churn risk (playbook CS-02). Score five signals per clinic over the last 4 weeks of data (the 4 weekly rows ending 2026-10-26):
1. seat utilization: average of weekly seat_utilization below 0.55;
2. P1 tickets: 3 or more in the 4 weeks;
3. CSAT: average avg_csat below 3.5;
4. visits trend: average weekly visits in the 4 weeks more than 20% below the average of the 12 weeks before;
5. invoices: any invoice due in the last 90 days paid more than 15 days late, or unpaid more than 15 days after its due date.
A clinic is "at risk" with 2 or more red signals. "Renewal risk" / "biggest renewal risk": at-risk clinics with their renewal date and ARR at stake (arr_usd), soonest renewal first; renewals within 90 days are urgent."""


def renewal_risk_sql(s):
    return f"""WITH w4 AS (
  SELECT tenant_id, AVG(seat_utilization) AS seat_utilization_4w, SUM(p1_tickets) AS p1_tickets_4w,
         AVG(avg_csat) AS csat_4w, AVG(visits) AS weekly_visits_4w
  FROM {s}.tenant_health_weekly
  WHERE week > date_sub((SELECT MAX(week) FROM {s}.tenant_health_weekly), 28)
  GROUP BY tenant_id),
prior12 AS (
  SELECT tenant_id, AVG(visits) AS weekly_visits_prior_12w
  FROM {s}.tenant_health_weekly
  WHERE week <= date_sub((SELECT MAX(week) FROM {s}.tenant_health_weekly), 28)
    AND week > date_sub((SELECT MAX(week) FROM {s}.tenant_health_weekly), 112)
  GROUP BY tenant_id),
invoices AS (
  SELECT tenant_id, MAX(COALESCE(days_late, datediff(DATE'{DATA_END}', due_date))) AS worst_days_late
  FROM {s}.revenue_monthly
  WHERE due_date > date_sub(DATE'{DATA_END}', 90) AND due_date <= DATE'{DATA_END}'
  GROUP BY tenant_id),
signals AS (
  SELECT c.clinic_name, c.region, c.renewal_date, c.arr_usd,
         ROUND(w4.seat_utilization_4w, 3) AS seat_utilization_4w, w4.p1_tickets_4w, ROUND(w4.csat_4w, 2) AS csat_4w,
         ROUND(w4.weekly_visits_4w / p.weekly_visits_prior_12w - 1, 3) AS visits_trend, i.worst_days_late,
         CAST(w4.seat_utilization_4w < 0.55 AS INT) + CAST(w4.p1_tickets_4w >= 3 AS INT)
           + CAST(COALESCE(w4.csat_4w < 3.5, false) AS INT) + CAST(w4.weekly_visits_4w < 0.8 * p.weekly_visits_prior_12w AS INT)
           + CAST(COALESCE(i.worst_days_late > 15, false) AS INT) AS red_signals
  FROM {s}.dim_clinic c JOIN w4 USING (tenant_id) JOIN prior12 p USING (tenant_id) LEFT JOIN invoices i USING (tenant_id))
SELECT clinic_name, region, renewal_date, datediff(renewal_date, DATE'{DATA_END}') AS days_to_renewal, arr_usd,
       red_signals, seat_utilization_4w, p1_tickets_4w, csat_4w, visits_trend, worst_days_late
FROM signals
WHERE red_signals >= 2
ORDER BY renewal_date"""


# Notebook 01 pastes these into the UI before the lesson (deliberately without the renewal-risk definition).
BASE_INSTRUCTIONS = f"""You are the clinic operations analyst for Kasterwell Health, a fictional virtual-care platform used by about 120 clinics.
Data covers {DATA_START} to {DATA_END}. Interpret "this month", "recent" and "this year" relative to the latest date in the data, not today's date.
Join tables on tenant_id. denial_rate = SUM(denied_claims) / SUM(decided_claims); pending claims are excluded.
Column values are lowercase: specialty is behavioral_health, primary_care, pediatrics, womens_health, urgent_care or dermatology; time_of_day is morning, afternoon or evening; region is West, Central or East.
Rates are 0-1 fractions; show them as percentages with one decimal. Lead with the number, then a one-line explanation."""

# ── Full instructions (production space) ─────────────────────────────────────────────────────────────────────────
INSTRUCTIONS = f"""You are the clinic operations analyst for Kasterwell Health, a fictional virtual-care platform used by about 120 clinics in three regions. Answer only from these tables; never estimate or invent a number.

DATA WINDOW
- Data covers {DATA_START} to {DATA_END}. The latest month is October 2026 and the latest health week starts {LATEST_WEEK}. Interpret "today", "this month", "this year", "recent" and "this week" relative to {DATA_END}, never the real current date.
- Defaults when the user gives no period: rankings and "how are we doing" questions use the last 3 months of data (Aug-Oct 2026); "this year" means Jan-Oct 2026, compared with Jan-Oct 2025 when asked about change; customer health uses the last 4 weekly rows.
- Claims: September and especially October 2026 still have pending claims, so claim and denial COUNTS for those months are incomplete. Compare denial RATES (pending excluded), not volumes, for recent months, and say so when you show volumes.
- Always state the period and filters you used in the answer (e.g. "Aug-Oct 2026, West, behavioral health").

METRICS (use the measures; never average a pre-computed rate)
- Denial rate = SUM(denied_claims) / SUM(decided_claims). Revenue at risk = SUM(denied_billed_usd).
- No-show rate = SUM(no_shows) / SUM(appointments) in appointment_access (or / SUM(appointments_booked) in clinic_daily_ops).
- Average wait = SUM(avg_wait_minutes * appointments) / SUM(appointments) (weighted); never AVG(avg_wait_minutes) across rows.
- Visits per clinician per month = SUM(visits) / number of clinician-months with active_days > 0. Visits per active day = SUM(visits) / SUM(active_days). Note minutes = SUM(avg_note_minutes * visits) / SUM(visits).
- 99215 share = claims with cpt_code 99215 / claims with cpt_category 'E/M established' (99212-99215). Network benchmark is about 10%.
- Network seat utilization = SUM(weekly_active_clinicians) / SUM(seats); for one clinic, average its weekly seat_utilization.
- Days late = days_late for paid invoices; for unpaid invoices, days from due_date to the data end (an unpaid invoice is still late).
- Rates are 0-1 fractions; show percentages with one decimal. Money in USD with no decimals.

DEFINITIONS
{RENEWAL_DEFINITION}
- Psychotherapy = CPT 90834 (45 min) and 90837 (60 min). CO-197 = prior authorization absent. Larkspan Health Plan (West) has required prior authorization for psychotherapy since 2026-04-01.
- AI Scribe impact: compare each adopting clinic's 3 months before its ai_scribe_enabled_date with the 3 months after, excluding the month it was enabled. SMS reminder impact: compare reminder_sent true vs false within the same segment.
- Respiratory surge (also "flu season", "last winter", "the surge"): Dec 2025-Feb 2026, the only surge in the data; compare with Sep-Nov 2025.
- "The increase", "the spike", "the drop" with no period given: find the month the change started from the monthly trend, then compare before vs after that month (for denials in 2026 that is April 2026). Never answer with the last 3 months vs the 3 before unless asked.

TABLE GUIDE
- dim_clinic: clinic attributes, plan, ARR, renewal date, add-on dates. Join everything on tenant_id.
- claims_monthly (+ dim_payer on payer_id): claims, denials, denial reasons, billed/paid, by payer and CPT.
- appointment_access: no-shows and waits by appointment type, visit mode, time of day, reminder_sent (monthly).
- clinic_daily_ops: daily bookings, visits, no-shows, waits, active clinicians.
- tenant_health_weekly: seat utilization, P1 tickets, CSAT, visits per week (customer health, CS-02).
- clinician_productivity_monthly: visits, active days, AI Scribe share, note minutes per clinician.
- revenue_monthly: Kasterwell's own invoices to clinics (paid / open / overdue, days late).

ACCESS AND HONESTY
- Results include only the clinics the user is allowed to see (row filters). Never describe a result as company-wide; say "across the N clinics you can see". If a named clinic returns no rows, say it may be outside the user's access; don't say it doesn't exist.
- These tables hold aggregates only, no patient-level data. If a question needs information that isn't here (e.g. why a payer changed a policy, what a protocol says), say so and point to the documents (payer bulletins, playbooks).
- If a question is ambiguous (which metric, which period), state the assumption you made, or ask.

ANSWER STYLE: lead with the number and the period, then one or two lines of explanation. For "what changed" questions show the before and after values and the month it changed."""

# ── Joins ────────────────────────────────────────────────────────────────────────────────────────────────────────
JOINS = [(t, "dim_clinic", "tenant_id", f"Clinic attributes (name, region, specialty, plan, ARR) for {t}")
         for t in ["appointment_access", "claims_monthly", "clinic_daily_ops", "clinician_productivity_monthly",
                   "revenue_monthly", "tenant_health_weekly"]]
JOINS.append(("claims_monthly", "dim_payer", "payer_id", "Payer type and operating regions for claims"))

# ── SQL snippets: (table, alias, sql, display name, synonyms, instruction) ──────────────────────────────────────
MEASURES = [
    ("claims_monthly", "denial_rate", "SUM(claims_monthly.denied_claims) / SUM(claims_monthly.decided_claims)", "denial rate",
     ["denial %", "denials rate"], "Share of decided claims that were denied; pending claims excluded"),
    ("claims_monthly", "denied_claims_total", "SUM(claims_monthly.denied_claims)", "denied claims", ["denials", "number of denials"],
     "Count of denied claims; incomplete for Sep-Oct 2026 (pending)"),
    ("claims_monthly", "revenue_at_risk_usd", "SUM(claims_monthly.denied_billed_usd)", "revenue at risk", ["denied dollars", "denied revenue"],
     "Billed value of denied claims"),
    ("claims_monthly", "claims_submitted", "SUM(claims_monthly.claims)", "claims submitted", ["claim volume"], "All submitted claims incl. pending"),
    ("claims_monthly", "paid_usd_total", "SUM(claims_monthly.paid_usd)", "paid amount", ["collections"], "Amount paid by payers"),
    ("claims_monthly", "share_99215",
     "SUM(CASE WHEN claims_monthly.cpt_code = '99215' THEN claims_monthly.claims ELSE 0 END) / "
     "SUM(CASE WHEN claims_monthly.cpt_category = 'E/M established' THEN claims_monthly.claims ELSE 0 END)",
     "99215 share", ["highest E/M level share", "upcoding rate", "level 5 share"],
     "Share of established-patient E/M visits billed at 99215; benchmark ~10% (COMP-04)"),
    ("appointment_access", "no_show_rate", "SUM(appointment_access.no_shows) / SUM(appointment_access.appointments)",
     "no-show rate", ["missed appointment rate", "DNA rate"], "No-shows / booked appointments"),
    ("appointment_access", "avg_wait_minutes_weighted",
     "SUM(appointment_access.avg_wait_minutes * appointment_access.appointments) / SUM(appointment_access.appointments)",
     "average wait (minutes)", ["wait time", "average wait"], "Appointment-weighted average wait"),
    ("appointment_access", "appointments_total", "SUM(appointment_access.appointments)", "appointments", ["bookings"], "Booked appointments"),
    ("clinic_daily_ops", "daily_no_show_rate", "SUM(clinic_daily_ops.no_shows) / SUM(clinic_daily_ops.appointments_booked)",
     "no-show rate (daily ops)", [], "Use for daily or weekly no-show trends"),
    ("clinic_daily_ops", "visits_completed_total", "SUM(clinic_daily_ops.visits_completed)", "completed visits", ["visits", "encounters"],
     "Completed visits"),
    ("clinic_daily_ops", "daily_avg_wait_weighted",
     "SUM(clinic_daily_ops.avg_wait_minutes * clinic_daily_ops.visits_completed) / SUM(clinic_daily_ops.visits_completed)",
     "average wait, daily ops (minutes)", [], "Visit-weighted average wait from daily data"),
    ("tenant_health_weekly", "network_seat_utilization",
     "SUM(tenant_health_weekly.weekly_active_clinicians) / SUM(tenant_health_weekly.seats)", "seat utilization (network)",
     ["overall utilization"], "Seat utilization across several clinics"),
    ("tenant_health_weekly", "p1_tickets_total", "SUM(tenant_health_weekly.p1_tickets)", "P1 tickets", ["P1s", "outages"], "Priority-1 tickets"),
    ("tenant_health_weekly", "csat_average", "AVG(tenant_health_weekly.avg_csat)", "CSAT", ["satisfaction"],
     "Average of weekly CSAT (CS-02 definition)"),
    ("clinician_productivity_monthly", "visits_per_clinician_month",
     "SUM(clinician_productivity_monthly.visits) / COUNT_IF(clinician_productivity_monthly.active_days > 0)",
     "visits per clinician per month", ["throughput", "productivity"], "Average monthly visits per working clinician"),
    ("clinician_productivity_monthly", "visits_per_active_day",
     "SUM(clinician_productivity_monthly.visits) / SUM(clinician_productivity_monthly.active_days)", "visits per active day", [],
     "Visits per day worked"),
    ("clinician_productivity_monthly", "note_minutes",
     "SUM(clinician_productivity_monthly.avg_note_minutes * clinician_productivity_monthly.visits) / SUM(clinician_productivity_monthly.visits)",
     "note minutes per visit", ["documentation time", "charting time"], "Visit-weighted note-writing time"),
    ("revenue_monthly", "invoiced_total_usd", "SUM(revenue_monthly.invoiced_usd)", "invoiced revenue", ["billings"], "Kasterwell invoices"),
    ("revenue_monthly", "overdue_usd",
     "SUM(CASE WHEN revenue_monthly.invoice_status = 'overdue' THEN revenue_monthly.invoiced_usd ELSE 0 END)",
     "overdue amount", ["past due", "AR overdue"], "Unpaid invoices past their due date"),
    ("dim_clinic", "arr_total_usd", "SUM(dim_clinic.arr_usd)", "ARR", ["annual recurring revenue"], "Annual recurring revenue"),
    ("dim_clinic", "clinic_count", "COUNT(DISTINCT dim_clinic.tenant_id)", "number of clinics", ["clinic count", "customers"],
     "Clinics (only those the user can see)"),
]
FILTERS = [
    ("claims_monthly", "claims_monthly.cpt_code IN ('90834', '90837')", "psychotherapy claims", ["therapy claims"],
     "Psychotherapy (45 and 60 min), the codes Larkspan's prior-auth rule covers"),
    ("claims_monthly", "claims_monthly.denial_reason_code = 'CO-197'", "prior-auth denials", ["missing prior authorization", "CO-197"],
     "Denials because prior authorization was absent"),
    ("claims_monthly", "claims_monthly.cpt_category = 'E/M established'", "established E/M visits", ["99212-99215"],
     "Established-patient office visits"),
    ("claims_monthly", "claims_monthly.month >= DATE'2026-04-01'", "since Larkspan prior auth", ["after April 2026"],
     "Since Larkspan's prior-auth rule took effect"),
    ("claims_monthly", f"claims_monthly.month >= DATE'{RECENT_3M}'", "last 3 months of claims", ["recent claims"], "Aug-Oct 2026"),
    ("appointment_access", "appointment_access.time_of_day = 'evening'", "evening appointments", ["after 5pm"], "Appointments at 17:00 or later"),
    ("appointment_access", "appointment_access.visit_mode = 'video'", "video visits", ["virtual", "telehealth"], "Video appointments"),
    ("appointment_access", f"appointment_access.month >= DATE'{RECENT_3M}'", "last 3 months of appointments", ["recent"], "Aug-Oct 2026"),
    ("appointment_access", "appointment_access.month BETWEEN DATE'2025-12-01' AND DATE'2026-02-01'", "respiratory surge",
     ["flu season", "winter surge"], "Dec 2025-Feb 2026"),
    ("tenant_health_weekly", f"tenant_health_weekly.week > DATE'{LATEST_WEEK}' - INTERVAL 28 DAYS", "last 4 weeks",
     ["rolling 4 weeks", "recent health"], "The 4 latest weekly rows (CS-02 window)"),
    ("dim_clinic", f"dim_clinic.renewal_date <= DATE'{DATA_END}' + INTERVAL 90 DAYS", "renewing in 90 days", ["upcoming renewals"],
     "Renewal within 90 days of the data end"),
    ("dim_clinic", "dim_clinic.ai_scribe_enabled_date IS NOT NULL", "AI Scribe clinics", ["scribe adopters"], "Clinics that turned on AI Scribe"),
]
EXPRESSIONS = [
    ("claims_monthly", "service_month", "CAST(claims_monthly.month AS DATE)", "service month", ["month"], "Month as a date"),
    ("claims_monthly", "service_year", "YEAR(claims_monthly.month)", "year", ["calendar year"], "Year for year-over-year"),
    ("appointment_access", "appointment_month", "CAST(appointment_access.month AS DATE)", "appointment month", [], "Month as a date"),
    ("revenue_monthly", "days_late_effective", f"COALESCE(revenue_monthly.days_late, DATEDIFF(DATE'{DATA_END}', revenue_monthly.due_date))",
     "days late (incl. unpaid)", ["days overdue", "days past due"],
     "Paid invoices: days paid after due date. Unpaid: days past due as of the data end. Use for 'late invoice' questions"),
    ("dim_clinic", "days_to_renewal", f"DATEDIFF(dim_clinic.renewal_date, DATE'{DATA_END}')", "days to renewal", [],
     "Days from the data end to the renewal date"),
]

# ── Example SQL: (question, tables needed, sql template with {s}, usage guidance) ───────────────────────────────
EXAMPLES = [
    ("Which clinics are our biggest renewal risk?", ["tenant_health_weekly", "dim_clinic", "revenue_monthly"],
     renewal_risk_sql, "Any churn-risk, renewal-risk or at-risk-account question (CS-02)."),
    ("Why did denials go up for West behavioral health in 2026?", ["claims_monthly"],
     lambda s: f"""SELECT payer_name,
       ROUND(SUM(CASE WHEN month < DATE'2026-04-01' THEN denied_claims END) / SUM(CASE WHEN month < DATE'2026-04-01' THEN decided_claims END), 3) AS denial_rate_jan_mar,
       ROUND(SUM(CASE WHEN month >= DATE'2026-04-01' THEN denied_claims END) / SUM(CASE WHEN month >= DATE'2026-04-01' THEN decided_claims END), 3) AS denial_rate_apr_oct,
       SUM(CASE WHEN month >= DATE'2026-04-01' THEN denied_claims END) AS denied_apr_oct
FROM {s}.claims_monthly
WHERE region = 'West' AND specialty = 'behavioral_health' AND month BETWEEN DATE'2026-01-01' AND DATE'2026-10-01'
GROUP BY payer_name
ORDER BY denial_rate_apr_oct - denial_rate_jan_mar DESC""",
     "'Why did denials change' questions: break the change down by payer (then CPT code and denial reason) before and after the change month."),
    ("What are the top denial reasons for West behavioral health since April 2026?", ["claims_monthly"],
     lambda s: f"""SELECT denial_reason_code, SUM(denied_claims) AS denied_claims, ROUND(SUM(denied_billed_usd), 0) AS revenue_at_risk_usd
FROM {s}.claims_monthly
WHERE region = 'West' AND specialty = 'behavioral_health' AND month >= DATE'2026-04-01' AND denied_claims > 0
GROUP BY denial_reason_code
ORDER BY denied_claims DESC""", "Denial reason breakdowns."),
    ("Monthly denial rate for Larkspan psychotherapy claims", ["claims_monthly"],
     lambda s: f"""SELECT CAST(month AS DATE) AS month, SUM(decided_claims) AS decided_claims, SUM(denied_claims) AS denied_claims,
       ROUND(SUM(denied_claims) / SUM(decided_claims), 3) AS denial_rate
FROM {s}.claims_monthly
WHERE payer_name = 'Larkspan Health Plan' AND cpt_code IN ('90834', '90837')
GROUP BY month
ORDER BY month""", "Monthly trend of one payer and code group."),
    ("Which clinics have the most revenue at risk from denials in the last 3 months?", ["claims_monthly", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, d.region, ROUND(SUM(c.denied_billed_usd), 0) AS revenue_at_risk_usd,
       ROUND(SUM(c.denied_claims) / SUM(c.decided_claims), 3) AS denial_rate
FROM {s}.claims_monthly c JOIN {s}.dim_clinic d USING (tenant_id)
WHERE c.month >= DATE'{RECENT_3M}'
GROUP BY d.clinic_name, d.region
ORDER BY revenue_at_risk_usd DESC
LIMIT 10""", "Clinic rankings on claims; state the period."),
    ("Do SMS reminders reduce evening no-shows for behavioral health video visits?", ["appointment_access"],
     lambda s: f"""SELECT reminder_sent, SUM(appointments) AS appointments, ROUND(SUM(no_shows) / SUM(appointments), 3) AS no_show_rate
FROM {s}.appointment_access
WHERE specialty = 'behavioral_health' AND time_of_day = 'evening' AND visit_mode = 'video'
GROUP BY reminder_sent
ORDER BY reminder_sent""", "Reminder impact: compare reminder_sent within the same segment."),
    ("Which clinics had the highest no-show rate in the last 3 months?", ["appointment_access", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, d.region, d.specialty, SUM(a.appointments) AS appointments,
       ROUND(SUM(a.no_shows) / SUM(a.appointments), 3) AS no_show_rate,
       ROUND(SUM(a.avg_wait_minutes * a.appointments) / SUM(a.appointments), 1) AS avg_wait_minutes,
       d.sms_reminders_enabled_date IS NOT NULL AS sms_reminders_on
FROM {s}.appointment_access a JOIN {s}.dim_clinic d USING (tenant_id)
WHERE a.month >= DATE'{RECENT_3M}'
GROUP BY d.clinic_name, d.region, d.specialty, d.sms_reminders_enabled_date
ORDER BY no_show_rate DESC
LIMIT 10""", "Access rankings: weighted waits, period Aug-Oct 2026 unless asked otherwise."),
    ("Which clinics had the worst video wait times during the respiratory surge?", ["appointment_access", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, d.region, d.specialty,
       ROUND(SUM(CASE WHEN a.month BETWEEN DATE'2025-09-01' AND DATE'2025-11-01' THEN a.avg_wait_minutes * a.appointments END)
           / SUM(CASE WHEN a.month BETWEEN DATE'2025-09-01' AND DATE'2025-11-01' THEN a.appointments END), 1) AS wait_sep_nov_2025,
       ROUND(SUM(CASE WHEN a.month BETWEEN DATE'2025-12-01' AND DATE'2026-02-01' THEN a.avg_wait_minutes * a.appointments END)
           / SUM(CASE WHEN a.month BETWEEN DATE'2025-12-01' AND DATE'2026-02-01' THEN a.appointments END), 1) AS wait_dec_feb
FROM {s}.appointment_access a JOIN {s}.dim_clinic d USING (tenant_id)
WHERE a.visit_mode = 'video'
GROUP BY d.clinic_name, d.region, d.specialty
ORDER BY wait_dec_feb DESC
LIMIT 10""", "Wait-time comparisons: always appointment-weighted."),
    ("Is any clinic's E/M coding pattern unusual?", ["claims_monthly", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, d.region, d.specialty,
       ROUND(SUM(CASE WHEN c.cpt_code = '99215' THEN c.claims ELSE 0 END) / SUM(CASE WHEN c.cpt_category = 'E/M established' THEN c.claims ELSE 0 END), 3) AS share_99215,
       SUM(CASE WHEN c.cpt_category = 'E/M established' THEN c.claims ELSE 0 END) AS established_visits
FROM {s}.claims_monthly c JOIN {s}.dim_clinic d USING (tenant_id)
WHERE c.month >= DATE'{RECENT_3M}'
GROUP BY d.clinic_name, d.region, d.specialty
HAVING established_visits > 0
ORDER BY share_99215 DESC
LIMIT 10""", "Coding-pattern questions; benchmark ~10%, COMP-04 reviews above 2.5x."),
    ("Which clinics have invoices more than 15 days late?", ["revenue_monthly", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, MAX(COALESCE(r.days_late, datediff(DATE'{DATA_END}', r.due_date))) AS worst_days_late,
       SUM(CASE WHEN r.invoice_status = 'overdue' THEN r.invoiced_usd ELSE 0 END) AS overdue_usd
FROM {s}.revenue_monthly r JOIN {s}.dim_clinic d USING (tenant_id)
WHERE r.due_date > date_sub(DATE'{DATA_END}', 90) AND r.due_date <= DATE'{DATA_END}'
GROUP BY d.clinic_name
HAVING worst_days_late > 15
ORDER BY worst_days_late DESC""", "Late invoices: unpaid invoices count as days past due as of the data end; default window is invoices due in the last 90 days."),
    ("What changed after clinics turned on AI Scribe?", ["clinician_productivity_monthly", "dim_clinic"],
     lambda s: f"""SELECT CASE WHEN p.month < date_trunc('MONTH', d.ai_scribe_enabled_date) THEN '3 months before' ELSE '3 months after' END AS period,
       ROUND(SUM(p.visits) / COUNT_IF(p.active_days > 0), 1) AS visits_per_clinician_month,
       ROUND(SUM(p.avg_note_minutes * p.visits) / SUM(p.visits), 2) AS note_minutes
FROM {s}.clinician_productivity_monthly p JOIN {s}.dim_clinic d USING (tenant_id)
WHERE d.ai_scribe_enabled_date IS NOT NULL
  AND p.month BETWEEN add_months(date_trunc('MONTH', d.ai_scribe_enabled_date), -3) AND add_months(date_trunc('MONTH', d.ai_scribe_enabled_date), 3)
  AND p.month <> date_trunc('MONTH', d.ai_scribe_enabled_date)
GROUP BY 1
ORDER BY 1 DESC""", "Before/after impact of an add-on within the same clinics."),
]

# ── Benchmarks: (question, tables needed, ground-truth sql with {s}) ───────────────────────────────────────────
BENCHMARKS = [
    ("How many clinics do we have in each region?", ["dim_clinic"],
     lambda s: f"SELECT region, COUNT(*) AS clinics FROM {s}.dim_clinic GROUP BY region ORDER BY clinics DESC"),
    ("How many West clinics are there in each specialty?", ["dim_clinic"],
     lambda s: f"SELECT specialty, COUNT(*) AS clinics FROM {s}.dim_clinic WHERE region = 'West' GROUP BY specialty ORDER BY clinics DESC"),
    ("Which clinics are our biggest renewal risk?", ["tenant_health_weekly", "dim_clinic", "revenue_monthly"], renewal_risk_sql),
    ("Which clinics are at risk of churning?", ["tenant_health_weekly", "dim_clinic", "revenue_monthly"], renewal_risk_sql),
    ("Which clinics renew in the next 90 days?", ["dim_clinic"],
     lambda s: f"SELECT clinic_name, renewal_date, arr_usd FROM {s}.dim_clinic WHERE renewal_date <= DATE'{DATA_END}' + INTERVAL 90 DAYS ORDER BY renewal_date"),
    ("What was Vellamy Family Health's seat utilization each month in 2026?", ["tenant_health_weekly"],
     lambda s: f"""SELECT date_trunc('MONTH', week) AS month, ROUND(AVG(seat_utilization), 3) AS seat_utilization
FROM {s}.tenant_health_weekly WHERE clinic_name = 'Vellamy Family Health' AND week >= DATE'2026-01-01' GROUP BY 1 ORDER BY 1"""),
    ("What was the denial rate for West behavioral health claims each month in 2026?", ["claims_monthly"],
     lambda s: f"""SELECT CAST(month AS DATE) AS month, ROUND(SUM(denied_claims) / SUM(decided_claims), 3) AS denial_rate
FROM {s}.claims_monthly WHERE region = 'West' AND specialty = 'behavioral_health' AND month >= DATE'2026-01-01' GROUP BY month ORDER BY month"""),
    ("Which payer had the highest denial rate for West behavioral health claims in 2026?", ["claims_monthly"],
     lambda s: f"""SELECT payer_name, ROUND(SUM(denied_claims) / SUM(decided_claims), 3) AS denial_rate, SUM(denied_claims) AS denied_claims
FROM {s}.claims_monthly WHERE region = 'West' AND specialty = 'behavioral_health' AND year(month) = 2026 GROUP BY payer_name ORDER BY denial_rate DESC LIMIT 1"""),
    ("What was Larkspan's psychotherapy denial rate before and after April 1, 2026?", ["claims_monthly"],
     lambda s: f"""SELECT month >= DATE'2026-04-01' AS after_april_2026, ROUND(SUM(denied_claims) / SUM(decided_claims), 3) AS denial_rate
FROM {s}.claims_monthly WHERE payer_name = 'Larkspan Health Plan' AND cpt_code IN ('90834', '90837')
  AND month BETWEEN DATE'2025-04-01' AND DATE'2026-10-01' GROUP BY 1 ORDER BY 1"""),
    ("What were the top denial reasons for West behavioral health claims since April 2026?", ["claims_monthly"],
     lambda s: f"""SELECT denial_reason_code, SUM(denied_claims) AS denied_claims FROM {s}.claims_monthly
WHERE region = 'West' AND specialty = 'behavioral_health' AND month >= DATE'2026-04-01' AND denied_claims > 0
GROUP BY denial_reason_code ORDER BY denied_claims DESC"""),
    ("Which West clinics had a denial rate above 10% in October 2026?", ["claims_monthly", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, ROUND(SUM(c.denied_claims) / SUM(c.decided_claims), 3) AS denial_rate
FROM {s}.claims_monthly c JOIN {s}.dim_clinic d USING (tenant_id) WHERE d.region = 'West' AND c.month = DATE'2026-10-01'
GROUP BY d.clinic_name HAVING SUM(c.denied_claims) / SUM(c.decided_claims) > 0.10 ORDER BY denial_rate DESC"""),
    ("Do SMS reminders reduce evening no-shows for behavioral health video visits?", ["appointment_access"],
     lambda s: f"""SELECT reminder_sent, ROUND(SUM(no_shows) / SUM(appointments), 3) AS no_show_rate FROM {s}.appointment_access
WHERE specialty = 'behavioral_health' AND time_of_day = 'evening' AND visit_mode = 'video' GROUP BY reminder_sent ORDER BY reminder_sent"""),
    ("What is the no-show rate by time of day over the last 3 months?", ["appointment_access"],
     lambda s: f"""SELECT time_of_day, ROUND(SUM(no_shows) / SUM(appointments), 3) AS no_show_rate FROM {s}.appointment_access
WHERE month >= DATE'{RECENT_3M}' GROUP BY time_of_day ORDER BY no_show_rate DESC"""),
    ("Which West clinics had the highest no-show rate in the last 3 months?", ["appointment_access", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, ROUND(SUM(a.no_shows) / SUM(a.appointments), 3) AS no_show_rate
FROM {s}.appointment_access a JOIN {s}.dim_clinic d USING (tenant_id) WHERE d.region = 'West' AND a.month >= DATE'{RECENT_3M}'
GROUP BY d.clinic_name ORDER BY no_show_rate DESC, d.clinic_name LIMIT 5"""),
    ("Which clinics had the longest average video wait in January 2026?", ["appointment_access", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, ROUND(SUM(a.avg_wait_minutes * a.appointments) / SUM(a.appointments), 1) AS avg_wait_minutes
FROM {s}.appointment_access a JOIN {s}.dim_clinic d USING (tenant_id) WHERE a.visit_mode = 'video' AND a.month = DATE'2026-01-01'
GROUP BY d.clinic_name ORDER BY avg_wait_minutes DESC, d.clinic_name LIMIT 6"""),
    ("What was the average video wait in the West each month from September 2025 to March 2026?", ["appointment_access"],
     lambda s: f"""SELECT CAST(month AS DATE) AS month, ROUND(SUM(avg_wait_minutes * appointments) / SUM(appointments), 1) AS avg_wait_minutes
FROM {s}.appointment_access WHERE region = 'West' AND visit_mode = 'video' AND month BETWEEN DATE'2025-09-01' AND DATE'2026-03-01'
GROUP BY month ORDER BY month"""),
    ("Which clinic has the highest 99215 share in the last 3 months?", ["claims_monthly", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, ROUND(SUM(CASE WHEN c.cpt_code = '99215' THEN c.claims ELSE 0 END) / SUM(CASE WHEN c.cpt_category = 'E/M established' THEN c.claims ELSE 0 END), 3) AS share_99215
FROM {s}.claims_monthly c JOIN {s}.dim_clinic d USING (tenant_id) WHERE c.month >= DATE'{RECENT_3M}'
GROUP BY d.clinic_name HAVING SUM(CASE WHEN c.cpt_category = 'E/M established' THEN c.claims ELSE 0 END) > 0 ORDER BY share_99215 DESC LIMIT 1"""),
    ("What was Halverro Urgent Care's 99215 share before and after February 2026?", ["claims_monthly", "dim_clinic"],
     lambda s: f"""SELECT c.month >= DATE'2026-02-01' AS from_feb_2026, ROUND(SUM(CASE WHEN c.cpt_code = '99215' THEN c.claims ELSE 0 END) / SUM(CASE WHEN c.cpt_category = 'E/M established' THEN c.claims ELSE 0 END), 3) AS share_99215
FROM {s}.claims_monthly c JOIN {s}.dim_clinic d USING (tenant_id) WHERE d.clinic_name = 'Halverro Urgent Care' GROUP BY 1 ORDER BY 1"""),
    ("What changed in visits per clinician and note time after clinics turned on AI Scribe?", ["clinician_productivity_monthly", "dim_clinic"],
     lambda s: [e for e in EXAMPLES if e[0].startswith("What changed after clinics turned on AI Scribe")][0][2](s)),
    ("Which clinics have an invoice more than 15 days late in the last 90 days?", ["revenue_monthly", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, MAX(COALESCE(r.days_late, datediff(DATE'{DATA_END}', r.due_date))) AS worst_days_late
FROM {s}.revenue_monthly r JOIN {s}.dim_clinic d USING (tenant_id)
WHERE r.due_date > date_sub(DATE'{DATA_END}', 90) AND r.due_date <= DATE'{DATA_END}'
GROUP BY d.clinic_name HAVING worst_days_late > 15 ORDER BY worst_days_late DESC"""),
    ("Which payer is driving the West behavioral health denial increase?", ["claims_monthly"],
     lambda s: f"""SELECT payer_name,
       ROUND(SUM(CASE WHEN month >= DATE'2026-04-01' THEN denied_claims END) / SUM(CASE WHEN month >= DATE'2026-04-01' THEN decided_claims END)
           - SUM(CASE WHEN month < DATE'2026-04-01' THEN denied_claims END) / SUM(CASE WHEN month < DATE'2026-04-01' THEN decided_claims END), 3) AS change_in_denial_rate
FROM {s}.claims_monthly WHERE region = 'West' AND specialty = 'behavioral_health' AND month BETWEEN DATE'2025-04-01' AND DATE'2026-10-01'
GROUP BY payer_name ORDER BY change_in_denial_rate DESC LIMIT 1"""),
    ("Which West clinics had the worst video wait times during the respiratory surge?", ["appointment_access", "dim_clinic"],
     lambda s: f"""SELECT d.clinic_name, ROUND(SUM(a.avg_wait_minutes * a.appointments) / SUM(a.appointments), 1) AS avg_wait_minutes
FROM {s}.appointment_access a JOIN {s}.dim_clinic d USING (tenant_id)
WHERE d.region = 'West' AND a.visit_mode = 'video' AND a.month BETWEEN DATE'2025-12-01' AND DATE'2026-02-01'
GROUP BY d.clinic_name ORDER BY avg_wait_minutes DESC, d.clinic_name LIMIT 6"""),
    ("What is our total ARR by plan tier?", ["dim_clinic"],
     lambda s: f"SELECT plan_tier, ROUND(SUM(arr_usd), 0) AS arr_usd FROM {s}.dim_clinic GROUP BY plan_tier ORDER BY arr_usd DESC"),
    ("Which payers had the biggest jump in denial rate in 2026 compared with 2025?", ["claims_monthly"],
     lambda s: f"""SELECT payer_name,
       ROUND(SUM(CASE WHEN year(month) = 2025 THEN denied_claims END) / SUM(CASE WHEN year(month) = 2025 THEN decided_claims END), 3) AS denial_rate_2025,
       ROUND(SUM(CASE WHEN year(month) = 2026 THEN denied_claims END) / SUM(CASE WHEN year(month) = 2026 THEN decided_claims END), 3) AS denial_rate_2026
FROM {s}.claims_monthly WHERE year(month) IN (2025, 2026) GROUP BY payer_name ORDER BY denial_rate_2026 - denial_rate_2025 DESC LIMIT 3"""),
]

SAMPLE_QUESTIONS = [
    "Which clinics are our biggest renewal risk?",
    "Why did denials go up for West behavioral health in 2026?",
    "Do SMS reminders reduce evening no-shows for behavioral health video visits?",
    "Which clinics had the worst video wait times during the respiratory surge?",
    "Is any clinic's E/M coding pattern unusual?",
    "What changed after clinics turned on AI Scribe?",
]


def _id(prefix, n):
    return f"{prefix}{n:031x}"


def build_space(schema, tables, table_comments=None):
    """serialized_space for a Genie Agent over `tables` in `schema` (only the parts those tables support)."""
    tables = set(tables)
    has = lambda need: set(need) <= tables
    out_tables = []
    for t in sorted(tables):
        desc, cols = TABLES[t]
        cfg = []
        for c in sorted(cols):
            d, syn, match, excl = cols[c]
            item = {"column_name": c}
            if excl:
                item["exclude"] = True
            else:
                item["description"] = [d]
                if syn:
                    item["synonyms"] = syn
                if match:
                    item["enable_entity_matching"] = True
            cfg.append(item)
        out_tables.append({"identifier": f"{schema}.{t}", "description": [(table_comments or {}).get(t, desc)], "column_configs": cfg})

    joins = [{"id": _id("4", i + 1), "left": {"identifier": f"{schema}.{l}", "alias": l}, "right": {"identifier": f"{schema}.{r}", "alias": r},
              "sql": [f"`{l}`.`{k}` = `{r}`.`{k}`", "--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--"], "comment": [c], "instruction": [c]}
             for i, (l, r, k, c) in enumerate(JOINS) if has([l, r])]

    def snip(kind, items, prefix):
        out = []
        for i, it in enumerate(items):
            if it[0] not in tables:
                continue
            if kind == "filters":
                _, sql, name, syn, ins = it
                out.append({"id": _id(prefix, i + 1), "sql": [sql], "display_name": name, "synonyms": syn, "instruction": [ins]})
            else:
                _, alias, sql, name, syn, ins = it
                out.append({"id": _id(prefix, i + 1), "alias": alias, "sql": [sql], "display_name": name, "synonyms": syn, "instruction": [ins]})
        return out

    examples = [{"id": _id("2", i + 1), "question": [q], "sql": [f(schema)], "usage_guidance": [g]}
                for i, (q, need, f, g) in enumerate(EXAMPLES) if has(need)]
    benchmarks = [{"id": _id("6", i + 1), "question": [q], "answer": [{"format": "SQL", "content": [f(schema)]}]}
                  for i, (q, need, f) in enumerate(BENCHMARKS) if has(need)]
    samples = [q for q in SAMPLE_QUESTIONS if any(e["question"][0] == q for e in examples) or q in [b["question"][0] for b in benchmarks]]
    return {
        "version": 2,
        "config": {"sample_questions": [{"id": _id("1", i + 1), "question": [q]} for i, q in enumerate(samples)]},
        "data_sources": {"tables": out_tables},
        "instructions": {
            "text_instructions": [{"id": _id("3", 1), "content": [INSTRUCTIONS]}],
            "example_question_sqls": examples,
            "join_specs": joins,
            "sql_snippets": {"filters": snip("filters", FILTERS, "5"), "expressions": snip("expressions", EXPRESSIONS, "7"),
                             "measures": snip("measures", MEASURES, "8")},
        },
        "benchmarks": {"questions": benchmarks},
    }


def benchmarks(schema, tables):
    """[(question, ground-truth sql)] for the tables a space has."""
    return [(q, f(schema)) for q, need, f in BENCHMARKS if set(need) <= set(tables)]


# ── clinic_snapshot: the agents' per-clinic tool (created by 00_setup) ─────────────────────
def clinic_snapshot_sql(s, fn_schema=None):
    """CREATE FUNCTION for <fn_schema>.clinic_snapshot over tables in <s>: metrics + the CS-02 verdict + network context."""
    return f"""CREATE OR REPLACE FUNCTION {fn_schema or s}.clinic_snapshot(clinic STRING COMMENT 'Clinic name, or part of it, e.g. Vellamy')
RETURNS TABLE (clinic_name STRING, region STRING, specialty STRING, plan_tier STRING, arr_usd DOUBLE, renewal_date DATE,
               days_to_renewal INT, addons STRING, visits_last_4w BIGINT, seat_utilization_4w DOUBLE, p1_tickets_4w BIGINT,
               csat_4w DOUBLE, visits_trend_vs_prior_12w DOUBLE, worst_invoice_days_late_90d INT,
               cs02_red_signals INT, cs02_red_signal_list STRING, cs02_at_risk BOOLEAN,
               denial_rate_3m DOUBLE, network_denial_rate_3m DOUBLE, no_show_rate_3m DOUBLE, network_no_show_rate_3m DOUBLE)
COMMENT 'One-row health snapshot for a clinic as of 2026-10-31: plan, renewal, the five CS-02 churn signals over the last 4 weeks with the verdict (at risk = 2+ red signals), and its 3-month denial and no-show rates next to the network average. Call it before answering any question about a specific clinic, and base any judgment of its health on cs02_at_risk.'
RETURN
  WITH c AS (SELECT * FROM {s}.dim_clinic WHERE lower(clinic_name) LIKE concat('%', lower(clinic), '%')),
  wk AS (SELECT MAX(week) AS d FROM {s}.tenant_health_weekly),
  h AS (SELECT tenant_id, SUM(visits) AS visits, AVG(visits) AS weekly_visits, AVG(seat_utilization) AS util,
               SUM(p1_tickets) AS p1, AVG(avg_csat) AS csat
        FROM {s}.tenant_health_weekly WHERE week > date_sub((SELECT d FROM wk), 28) GROUP BY tenant_id),
  p12 AS (SELECT tenant_id, AVG(visits) AS weekly_visits FROM {s}.tenant_health_weekly
          WHERE week <= date_sub((SELECT d FROM wk), 28) AND week > date_sub((SELECT d FROM wk), 112) GROUP BY tenant_id),
  inv AS (SELECT tenant_id, MAX(COALESCE(days_late, datediff(DATE'{DATA_END}', due_date))) AS worst_late
          FROM {s}.revenue_monthly WHERE due_date > date_sub(DATE'{DATA_END}', 90) AND due_date <= DATE'{DATA_END}' GROUP BY tenant_id),
  cl AS (SELECT tenant_id, SUM(denied_claims) AS denied, SUM(decided_claims) AS decided
         FROM {s}.claims_monthly WHERE month >= DATE'{RECENT_3M}' GROUP BY tenant_id),
  ap AS (SELECT tenant_id, SUM(no_shows) AS ns, SUM(appointments) AS appts
         FROM {s}.appointment_access WHERE month >= DATE'{RECENT_3M}' GROUP BY tenant_id),
  net AS (SELECT (SELECT SUM(denied) / SUM(decided) FROM cl) AS denial_rate, (SELECT SUM(ns) / SUM(appts) FROM ap) AS no_show_rate),
  sig AS (SELECT c.*, h.visits, h.util, h.p1, h.csat, h.weekly_visits / p12.weekly_visits - 1 AS trend, inv.worst_late,
                 cl.denied / NULLIF(cl.decided, 0) AS denial_rate, ap.ns / NULLIF(ap.appts, 0) AS no_show_rate,
                 filter(array(CASE WHEN h.util < 0.55 THEN 'seat utilization below 55%' END,
                              CASE WHEN h.p1 >= 3 THEN '3+ P1 tickets' END,
                              CASE WHEN h.csat < 3.5 THEN 'CSAT below 3.5' END,
                              CASE WHEN h.weekly_visits < 0.8 * p12.weekly_visits THEN 'visits down more than 20%' END,
                              CASE WHEN inv.worst_late > 15 THEN 'invoice more than 15 days late' END), x -> x IS NOT NULL) AS red
          FROM c LEFT JOIN h USING (tenant_id) LEFT JOIN p12 USING (tenant_id) LEFT JOIN inv USING (tenant_id)
                 LEFT JOIN cl USING (tenant_id) LEFT JOIN ap USING (tenant_id))
  SELECT clinic_name, region, specialty, plan_tier, arr_usd, renewal_date, datediff(renewal_date, DATE'{DATA_END}'), addons,
         visits, ROUND(util, 3), p1, ROUND(csat, 2), ROUND(trend, 3), worst_late,
         size(red), array_join(red, '; '), size(red) >= 2,
         ROUND(denial_rate, 3), ROUND((SELECT denial_rate FROM net), 3), ROUND(no_show_rate, 3), ROUND((SELECT no_show_rate FROM net), 3)
  FROM sig
  ORDER BY lower(clinic_name) = lower(clinic) DESC, lower(clinic_name) LIKE concat(lower(clinic), '%') DESC, clinic_name
  LIMIT 5"""

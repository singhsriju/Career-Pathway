import pandas as pd
import numpy as np
import os

def generate_dataset(n=2000, seed=2024):
    np.random.seed(seed)
    N = n

    persona_names = ["Confused Explorer","Focused Achiever","Career Switcher",
                     "Budget Conscious Learner","Parent Driven Student"]
    persona_probs = [0.25, 0.20, 0.15, 0.25, 0.15]
    persona_raw = np.random.choice(range(5), N, p=persona_probs)
    persona_label = np.array(persona_names)[persona_raw]

    def pc(options, weights_by_persona):
        result = np.empty(N, dtype=object)
        for pid, weights in enumerate(weights_by_persona):
            mask = persona_raw == pid
            if mask.sum():
                w = np.array(weights, dtype=float); w /= w.sum()
                result[mask] = np.random.choice(options, mask.sum(), p=w)
        return result

    def pc_int(low, high, means_by_persona, std=0.9):
        result = np.zeros(N, dtype=int)
        for pid, mean in enumerate(means_by_persona):
            mask = persona_raw == pid
            if mask.sum():
                raw = np.random.normal(mean, std, mask.sum())
                result[mask] = np.clip(np.round(raw), low, high).astype(int)
        return result

    city_tier = pc(["Tier 1","Tier 2","Tier 3"],
        [[0.40,0.38,0.22],[0.45,0.38,0.17],[0.35,0.42,0.23],[0.18,0.50,0.32],[0.20,0.48,0.32]])

    gender = pc(["Male","Female","Other/Prefer not to say"],
        [[0.52,0.44,0.04],[0.50,0.46,0.04],[0.55,0.42,0.03],[0.53,0.43,0.04],[0.51,0.45,0.04]])

    education_level = pc(
        ["Grade 8-10","Grade 11-12","Undergraduate","Postgraduate"],
        [[0.30,0.35,0.28,0.07],[0.08,0.22,0.45,0.25],[0.05,0.20,0.48,0.27],
         [0.22,0.32,0.35,0.11],[0.25,0.35,0.30,0.10]])

    age_map = {"Grade 8-10":(13,16),"Grade 11-12":(16,19),"Undergraduate":(18,23),"Postgraduate":(22,28)}
    age = np.array([int(np.random.uniform(*age_map[e])) for e in education_level])

    stream = pc(["Science","Commerce","Arts","Vocational/Other"],
        [[0.35,0.25,0.25,0.15],[0.48,0.28,0.16,0.08],[0.42,0.28,0.20,0.10],
         [0.38,0.25,0.22,0.15],[0.38,0.24,0.22,0.16]])

    institution_type = pc(
        ["Government/Aided","Private","Central School","Online/Distance","Not enrolled"],
        [[0.38,0.42,0.10,0.07,0.03],[0.25,0.55,0.10,0.08,0.02],[0.32,0.45,0.10,0.10,0.03],
         [0.45,0.35,0.08,0.10,0.02],[0.42,0.40,0.10,0.06,0.02]])

    family_income = pc(
        ["Below Rs2L","Rs2L-Rs5L","Rs5L-Rs10L","Rs10L-Rs20L","Above Rs20L"],
        [[0.20,0.35,0.28,0.12,0.05],[0.08,0.22,0.35,0.25,0.10],[0.15,0.30,0.32,0.18,0.05],
         [0.25,0.38,0.25,0.10,0.02],[0.22,0.35,0.28,0.12,0.03]])
    family_income_score = np.array([{"Below Rs2L":1,"Rs2L-Rs5L":2,"Rs5L-Rs10L":3,"Rs10L-Rs20L":4,"Above Rs20L":5}[v] for v in family_income])

    prior_paid_course = pc(
        ["Yes currently","Yes before","No free only","Parents pay"],
        [[0.08,0.12,0.60,0.20],[0.38,0.30,0.22,0.10],[0.22,0.28,0.38,0.12],
         [0.10,0.15,0.58,0.17],[0.15,0.12,0.35,0.38]])
    prior_paid_score = np.array([{"Yes currently":4,"Yes before":3,"No free only":1,"Parents pay":2}[v] for v in prior_paid_course])

    exam_prep = pc(
        ["JEE","NEET","CUET","CAT/MBA","UPSC","CLAT","NDA","Design","None","Not sure"],
        [[0.08,0.08,0.12,0.05,0.06,0.03,0.04,0.03,0.28,0.23],
         [0.18,0.12,0.15,0.12,0.10,0.05,0.05,0.05,0.12,0.06],
         [0.12,0.10,0.12,0.12,0.10,0.05,0.06,0.05,0.18,0.10],
         [0.06,0.06,0.10,0.08,0.08,0.03,0.05,0.03,0.32,0.19],
         [0.10,0.10,0.12,0.06,0.08,0.04,0.06,0.03,0.25,0.16]])

    acad_performance_score = pc_int(1,5,[2.8,4.1,3.2,3.0,2.9],std=0.8)
    acad_performance_label = np.array(["Struggling","Below Average","Average","Good","Excellent"])[acad_performance_score-1]

    self_learning_hrs = pc(
        ["Less than 1hr","1-3hrs","3-5hrs","5-10hrs","More than 10hrs"],
        [[0.20,0.38,0.28,0.10,0.04],[0.05,0.18,0.32,0.30,0.15],[0.12,0.30,0.32,0.18,0.08],
         [0.18,0.35,0.30,0.13,0.04],[0.22,0.38,0.28,0.09,0.03]])
    self_learning_score = np.array([{"Less than 1hr":1,"1-3hrs":2,"3-5hrs":3,"5-10hrs":4,"More than 10hrs":5}[v] for v in self_learning_hrs])

    career_clarity_stage = pc(
        ["No idea where to begin","Have interests, dont know career match",
         "Know what I want, not how to get there","Parents decided - not sure its right",
         "Chosen path, need plan","On path but feel I chose wrongly"],
        [[0.38,0.32,0.12,0.10,0.05,0.03],[0.03,0.08,0.28,0.04,0.45,0.12],
         [0.05,0.15,0.20,0.12,0.18,0.30],[0.18,0.28,0.25,0.08,0.15,0.06],
         [0.10,0.12,0.10,0.45,0.18,0.05]])

    confusion_level_score = pc_int(1,5,[4.2,1.8,3.2,3.0,3.5],std=0.7)
    confusion_level_label = np.array(["Very Low","Low","Medium","High","Very High"])[confusion_level_score-1]

    worry_frequency_score = pc_int(1,5,[4.0,2.2,3.5,3.2,3.8],std=0.8)
    worry_frequency_label = np.array(["Never","Rarely","Sometimes","Often","Almost Every Day"])[worry_frequency_score-1]

    family_decision_authority = pc(
        ["I decide completely","Mostly me parents input","Equal joint decision",
         "Mostly parents I follow","Parents already decided"],
        [[0.08,0.18,0.28,0.30,0.16],[0.42,0.35,0.15,0.06,0.02],[0.25,0.30,0.28,0.12,0.05],
         [0.15,0.25,0.32,0.20,0.08],[0.02,0.05,0.12,0.38,0.43]])
    family_decision_score = np.array([{"I decide completely":5,"Mostly me parents input":4,"Equal joint decision":3,"Mostly parents I follow":2,"Parents already decided":1}[v] for v in family_decision_authority])

    trusted_info_source = pc(
        ["YouTube/Instagram","Parents/relatives","School counselor",
         "Friends in career","Official websites","Career apps","Dont trust any"],
        [[0.28,0.25,0.15,0.12,0.08,0.05,0.07],[0.18,0.10,0.12,0.20,0.15,0.18,0.07],
         [0.22,0.15,0.15,0.18,0.12,0.12,0.06],[0.30,0.20,0.12,0.15,0.10,0.08,0.05],
         [0.15,0.38,0.18,0.10,0.10,0.05,0.04]])

    prior_platform_use = pc(
        ["Yes helpful","Yes not helpful enough","No but looked","No didnt know"],
        [[0.05,0.12,0.40,0.43],[0.28,0.30,0.30,0.12],[0.12,0.28,0.38,0.22],
         [0.06,0.14,0.38,0.42],[0.08,0.15,0.35,0.42]])

    biggest_challenge = pc(
        ["Too many options","Dont know strengths","Family pressure",
         "Lack of info new careers","Wrong college choice","Job availability worry","No guidance"],
        [[0.22,0.28,0.12,0.15,0.10,0.08,0.05],[0.08,0.10,0.05,0.18,0.22,0.25,0.12],
         [0.10,0.15,0.18,0.20,0.15,0.15,0.07],[0.12,0.18,0.10,0.15,0.18,0.20,0.07],
         [0.10,0.12,0.30,0.15,0.15,0.10,0.08]])

    emotional_pain_type = pc(
        ["Overwhelmed by options","Lacks self-awareness","Family conflict",
         "Information gap","Execution gap","Financial anxiety"],
        [[0.35,0.30,0.15,0.12,0.05,0.03],[0.05,0.10,0.05,0.20,0.40,0.20],
         [0.08,0.12,0.30,0.20,0.18,0.12],[0.12,0.15,0.10,0.15,0.15,0.33],
         [0.10,0.10,0.35,0.18,0.15,0.12]])

    risk_tolerance = pc(
        ["Stable job","High-risk reward","Depends on factors"],
        [[0.55,0.22,0.23],[0.28,0.52,0.20],[0.35,0.38,0.27],[0.60,0.18,0.22],[0.58,0.20,0.22]])
    risk_tolerance_score = np.array([{"Stable job":1,"High-risk reward":3,"Depends on factors":2}[v] for v in risk_tolerance])

    motivation_type = pc(
        ["Mostly financial","Balanced both","Mostly personal growth"],
        [[0.32,0.40,0.28],[0.20,0.32,0.48],[0.25,0.38,0.37],[0.38,0.38,0.24],[0.30,0.42,0.28]])
    motivation_score = pc_int(1,5,[2.5,4.0,3.2,2.8,3.0],std=0.8)

    locus_of_control_score = pc_int(1,5,[2.5,4.3,3.4,3.0,2.8],std=0.7)
    locus_of_control_label = np.array(["External","Mostly External","Balanced","Mostly Internal","Internal"])[locus_of_control_score-1]

    career_stage = pc(
        ["Exploration","Decision Making","Planning","Execution","Transition"],
        [[0.55,0.30,0.10,0.03,0.02],[0.05,0.15,0.35,0.40,0.05],[0.10,0.20,0.25,0.18,0.27],
         [0.35,0.32,0.22,0.08,0.03],[0.28,0.35,0.22,0.10,0.05]])

    career_domains_pool = ["Technology","Medicine","Business","Law","Design","Teaching",
                           "Science Research","Defence","Media","Finance","Sports","Social Work"]
    domain_w = {
        "Science":    [0.28,0.22,0.10,0.05,0.07,0.05,0.12,0.04,0.03,0.02,0.01,0.01],
        "Commerce":   [0.10,0.03,0.28,0.08,0.06,0.05,0.03,0.03,0.06,0.22,0.03,0.03],
        "Arts":       [0.05,0.03,0.10,0.18,0.15,0.12,0.06,0.08,0.12,0.04,0.04,0.03],
        "Vocational/Other":[0.22,0.12,0.12,0.05,0.10,0.08,0.05,0.10,0.05,0.05,0.04,0.02],
    }
    d1,d2,d3=[],[],[]
    for s in stream:
        w = np.array(domain_w.get(s,[1/12]*12)); w /= w.sum()
        c = np.random.choice(career_domains_pool,3,replace=False,p=w)
        d1.append(c[0]); d2.append(c[1]); d3.append(c[2])

    learning_preference = pc(
        ["Watching videos","Reading articles","Talking to mentor",
         "Hands-on projects","Quizzes interactive","Live workshops"],
        [[0.38,0.15,0.20,0.10,0.10,0.07],[0.22,0.18,0.20,0.22,0.10,0.08],
         [0.28,0.18,0.22,0.18,0.08,0.06],[0.35,0.18,0.18,0.12,0.12,0.05],
         [0.30,0.15,0.25,0.12,0.10,0.08]])

    primary_device = pc(
        ["Smartphone Android","Smartphone iOS","Laptop/Desktop","Shared family device"],
        [[0.60,0.08,0.22,0.10],[0.42,0.18,0.35,0.05],[0.52,0.12,0.30,0.06],
         [0.62,0.08,0.20,0.10],[0.58,0.08,0.18,0.16]])

    language_preference = pc(
        ["English only","Hindi","Regional language","English Hindi mix","No preference"],
        [[0.18,0.32,0.22,0.20,0.08],[0.35,0.22,0.15,0.22,0.06],[0.25,0.28,0.20,0.20,0.07],
         [0.15,0.35,0.25,0.18,0.07],[0.12,0.35,0.28,0.18,0.07]])

    feat_base = {
        "feat_career_match":[3.5,4.5,4.0,3.8,3.6],
        "feat_roadmap":[3.8,4.6,4.2,3.9,3.8],
        "feat_job_market":[3.2,4.4,3.8,3.5,3.3],
        "feat_chatbot":[3.4,4.0,3.7,3.5,3.4],
        "feat_college_rec":[3.6,4.1,3.8,3.7,3.8],
        "feat_mock_interview":[3.0,4.3,3.8,3.2,3.1],
        "feat_scholarship":[4.0,3.8,3.9,4.3,4.1],
        "feat_peer_bench":[3.2,3.8,3.5,3.4,3.2],
    }
    feat_scores = {}
    for fname, means in feat_base.items():
        fs = np.zeros(N, dtype=int)
        for pid, mean in enumerate(means):
            mask = persona_raw == pid
            if mask.sum():
                raw = np.random.normal(mean, 0.85, mask.sum())
                fs[mask] = np.clip(np.round(raw), 1, 5).astype(int)
        feat_scores[fname] = fs

    longitudinal_consent = pc(["Yes","No"],[[0.50,0.50],[0.75,0.25],[0.65,0.35],[0.55,0.45],[0.55,0.45]])

    wtp_band_opts = ["Free only","Up to Rs99","Rs100-Rs299","Rs300-Rs499","Rs500-Rs999","Rs1000+"]
    wtp_monthly_band = pc(wtp_band_opts,
        [[0.42,0.30,0.18,0.07,0.02,0.01],[0.08,0.15,0.28,0.30,0.15,0.04],
         [0.18,0.25,0.30,0.18,0.07,0.02],[0.35,0.32,0.22,0.08,0.02,0.01],
         [0.15,0.20,0.28,0.25,0.10,0.02]])

    wtp_map = {"Free only":0,"Up to Rs99":1,"Rs100-Rs299":3,"Rs300-Rs499":5,"Rs500-Rs999":7,"Rs1000+":9}
    wtp_score_raw = np.array([wtp_map[v] for v in wtp_monthly_band])
    wtp_score_continuous = np.clip(wtp_score_raw + np.random.uniform(-0.4,0.4,N),0,10).round(1)

    pricing_model_pref = pc(
        ["Completely free","Free one-time report","Monthly subscription","Annual plan","School pays","Pay per session"],
        [[0.45,0.28,0.12,0.08,0.05,0.02],[0.05,0.18,0.35,0.28,0.10,0.04],
         [0.18,0.28,0.25,0.18,0.08,0.03],[0.40,0.30,0.15,0.08,0.05,0.02],
         [0.12,0.15,0.18,0.12,0.38,0.05]])

    payment_trigger = pc(
        ["Friend recommendation","Teacher recommendation","Free trial","Outcome data","School subscribed","Affordable price","No trigger"],
        [[0.12,0.15,0.20,0.12,0.10,0.22,0.09],[0.20,0.18,0.22,0.25,0.08,0.05,0.02],
         [0.15,0.15,0.25,0.22,0.08,0.12,0.03],[0.10,0.12,0.18,0.10,0.08,0.35,0.07],
         [0.08,0.20,0.15,0.12,0.30,0.12,0.03]])

    will_pay_raw = np.where(wtp_monthly_band == "Free only", 0, 1)

    will_use_prob = np.zeros(N)
    for i in range(N):
        p = 0.40
        if confusion_level_score[i] >= 3: p += 0.12
        if locus_of_control_score[i] >= 4: p += 0.10
        if prior_platform_use[i] == "Yes not helpful enough": p += 0.12
        if prior_platform_use[i] == "No but looked": p += 0.07
        if feat_scores["feat_roadmap"][i] >= 4: p += 0.08
        if feat_scores["feat_career_match"][i] >= 4: p += 0.06
        if self_learning_score[i] >= 3: p += 0.05
        if prior_paid_score[i] >= 3: p += 0.06
        pid = persona_raw[i]
        p += [0.0,0.18,0.10,-0.05,0.05][pid]
        will_use_prob[i] = min(p, 0.97)

    will_use_raw = (np.random.rand(N) < will_use_prob).astype(int)
    noise_idx = np.random.choice(N, int(N*0.02), replace=False)
    will_use_raw[noise_idx] = 1 - will_use_raw[noise_idx]
    will_use_platform = np.where(will_use_raw==1,"Yes","No")

    outlier_idx = np.random.choice(N, int(N*0.03), replace=False)
    for i in outlier_idx[:len(outlier_idx)//2]:
        wtp_score_continuous[i] = np.random.uniform(7,9.5)
    for i in outlier_idx[len(outlier_idx)//2:]:
        confusion_level_score[i] = 5

    contra_idx = np.random.choice(N, int(N*0.02), replace=False)
    for i in contra_idx:
        wtp_score_continuous[i] = np.random.uniform(6,8)
        will_pay_raw[i] = 0

    df = pd.DataFrame({
        "respondent_id": [f"R{str(i+1).zfill(4)}" for i in range(N)],
        "persona_label": persona_label,
        "persona_id": persona_raw,
        "A_city_tier": city_tier,
        "A_gender": gender,
        "A_age": age,
        "A_education_level": education_level,
        "A_stream": stream,
        "A_institution_type": institution_type,
        "A_family_income_band": family_income,
        "A_family_income_score": family_income_score,
        "B_prior_paid_course": prior_paid_course,
        "B_prior_paid_score": prior_paid_score,
        "B_exam_prep": exam_prep,
        "B_acad_performance_score": acad_performance_score,
        "B_acad_performance_label": acad_performance_label,
        "B_self_learning_hrs": self_learning_hrs,
        "B_self_learning_score": self_learning_score,
        "C_career_clarity_stage": career_clarity_stage,
        "C_confusion_level_score": confusion_level_score,
        "C_confusion_level_label": confusion_level_label,
        "C_worry_frequency_score": worry_frequency_score,
        "C_worry_frequency_label": worry_frequency_label,
        "C_family_decision_authority": family_decision_authority,
        "C_family_decision_score": family_decision_score,
        "C_trusted_info_source": trusted_info_source,
        "C_prior_platform_use": prior_platform_use,
        "C_biggest_challenge": biggest_challenge,
        "C_emotional_pain_type": emotional_pain_type,
        "D_risk_tolerance": risk_tolerance,
        "D_risk_tolerance_score": risk_tolerance_score,
        "D_motivation_type": motivation_type,
        "D_motivation_score": motivation_score,
        "D_locus_of_control_score": locus_of_control_score,
        "D_locus_of_control_label": locus_of_control_label,
        "D_career_stage": career_stage,
        "E_career_domain_1": d1,
        "E_career_domain_2": d2,
        "E_career_domain_3": d3,
        "E_learning_preference": learning_preference,
        "E_feat_career_match": feat_scores["feat_career_match"],
        "E_feat_roadmap": feat_scores["feat_roadmap"],
        "E_feat_job_market": feat_scores["feat_job_market"],
        "E_feat_chatbot": feat_scores["feat_chatbot"],
        "E_feat_college_rec": feat_scores["feat_college_rec"],
        "E_feat_mock_interview": feat_scores["feat_mock_interview"],
        "E_feat_scholarship": feat_scores["feat_scholarship"],
        "E_feat_peer_bench": feat_scores["feat_peer_bench"],
        "E_primary_device": primary_device,
        "E_language_preference": language_preference,
        "E_longitudinal_consent": longitudinal_consent,
        "F_wtp_monthly_band": wtp_monthly_band,
        "F_wtp_score_continuous": wtp_score_continuous,
        "F_pricing_model_pref": pricing_model_pref,
        "F_payment_trigger": payment_trigger,
        "F_will_pay_binary": will_pay_raw,
        "F_will_use_platform": will_use_platform,
        "F_will_use_binary": will_use_raw,
    })

    non_critical = ["B_exam_prep","C_prior_platform_use","E_learning_preference",
                    "E_language_preference","E_feat_peer_bench","E_feat_mock_interview",
                    "D_career_stage","C_trusted_info_source","B_self_learning_hrs","E_longitudinal_consent"]
    for col in non_critical:
        missing_idx = np.random.choice(N, int(N*0.005), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def get_or_generate_dataset(filepath="career_dataset.csv"):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    df = generate_dataset()
    df.to_csv(filepath, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("career_dataset.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df["persona_label"].value_counts())

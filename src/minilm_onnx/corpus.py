"""Small query/document set for the retrieval-agreement parity check.

Written for this repo (not drawn from HealthMate-AI's corpus). It only
needs to produce realistic token lengths and a non-trivial ranking so we
can check that ONNX variants rank documents the same way PyTorch does.
It is not a quality benchmark for the model.
"""

DOCS = [
    "Type 2 diabetes is managed with diet, exercise, and medications such as metformin.",
    "Metformin lowers hepatic glucose production and improves insulin sensitivity.",
    "Hypertension is diagnosed when blood pressure is persistently at or above 130/80 mmHg.",
    "ACE inhibitors can cause a dry cough; ARBs are a common alternative.",
    "Asthma exacerbations are treated with short-acting beta-agonists and systemic corticosteroids.",
    "Inhaled corticosteroids are the foundation of long-term asthma control.",
    "Community-acquired pneumonia often presents with fever, cough, and focal crackles.",
    "Chest radiography confirms consolidation in suspected bacterial pneumonia.",
    "Iron deficiency anemia causes fatigue, pallor, and a low mean corpuscular volume.",
    "Vitamin B12 deficiency can produce macrocytic anemia and peripheral neuropathy.",
    "Migraine headaches are often unilateral, pulsating, and accompanied by photophobia.",
    "Triptans are used for acute migraine relief but are avoided in coronary artery disease.",
    "Atrial fibrillation increases stroke risk; anticoagulation is guided by the CHA2DS2-VASc score.",
    "Warfarin requires regular INR monitoring and has many food and drug interactions.",
    "Direct oral anticoagulants need less monitoring than warfarin.",
    "Hypothyroidism presents with weight gain, cold intolerance, and elevated TSH.",
    "Levothyroxine should be taken on an empty stomach, separate from calcium and iron.",
    "Hyperthyroidism may cause palpitations, heat intolerance, tremor, and weight loss.",
    "Chronic kidney disease is staged by estimated glomerular filtration rate and albuminuria.",
    "NSAIDs can worsen renal function, especially with dehydration or ACE inhibitor use.",
    "Urinary tract infections commonly present with dysuria, frequency, and urgency.",
    "Nitrofurantoin is a first-line option for uncomplicated cystitis in women.",
    "Major depressive disorder involves low mood or anhedonia for at least two weeks.",
    "SSRIs may take four to six weeks to reach full antidepressant effect.",
    "Generalized anxiety disorder features excessive worry most days for six months.",
    "Osteoarthritis causes joint pain that worsens with activity and improves with rest.",
    "Rheumatoid arthritis causes symmetric small-joint swelling and morning stiffness.",
    "Gout flares are triggered by uric acid crystal deposition, often in the big toe.",
    "Allopurinol lowers serum uric acid for long-term gout prevention.",
    "Gastroesophageal reflux disease causes heartburn and is treated with proton pump inhibitors.",
    "Helicobacter pylori infection is a major cause of peptic ulcer disease.",
    "Celiac disease is an immune reaction to gluten that damages the small intestine.",
    "Seasonal influenza vaccination is recommended annually for most adults.",
    "Tetanus boosters are recommended every ten years for adults.",
    "Sepsis requires early antibiotics, fluid resuscitation, and lactate monitoring.",
    "Heart failure with reduced ejection fraction benefits from beta-blockers and SGLT2 inhibitors.",
    "Loop diuretics relieve congestion in acute decompensated heart failure.",
    "COPD is confirmed by spirometry showing a post-bronchodilator FEV1/FVC below 0.7.",
    "Smoking cessation is the most effective intervention to slow COPD progression.",
    "Stroke symptoms include sudden facial droop, arm weakness, and speech difficulty.",
]

QUERIES = [
    "first-line medication for type 2 diabetes",
    "why does my blood pressure pill make me cough",
    "treatment for an asthma attack",
    "signs of pneumonia on a chest x-ray",
    "causes of feeling tired with small red blood cells",
    "medicine for a migraine that is not safe with heart disease",
    "blood thinner that needs INR checks",
    "how to take thyroid medication correctly",
    "painkillers that can hurt the kidneys",
    "antibiotic for a bladder infection",
    "how long until antidepressants start working",
    "painful swollen big toe from uric acid",
    "treatment for heartburn",
    "how is COPD diagnosed",
    "warning signs of a stroke",
]

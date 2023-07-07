from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-2B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-2B")

text = """CREATE TABLE experiment.resident_population_type_citizenship_category_gender_age (
    uid              NUMBER,
    spatialunit_uid  CHAR,
    year             NUMBER,
    population_type  FLOAT,
    gender           CHAR,
    age_group        CHAR,
    amount           NUMBER,
)

CREATE TABLE experiment.spatial_unit (
    spatialunit_uid        CHAR,
    spatialunit_current_id NUMBER,
    spatialunit_ontology   CHAR,
    name                   CHAR,
    name_de                CHAR,
    name_fr                CHAR,
    name_it                CHAR,
    country                CHAR,
    canton                 CHAR,
    district               CHAR,
    municipal              CHAR,
    residence_area         CHAR,
    neighborhood           CHAR,
    region                 CHAR,
    zone                   CHAR,
    spatialunit_hist_id    NUMBER,
    canton_hist_id         NUMBER,
    district_hist_id       NUMBER,
    valid_from             DATE,
    valid_until            DATE,
)


-- Using valid SQLite, answer the following questions for the tables provided above.

-- which spatialunit_uid had the highest amount for year = 2019 and population_type = 'Non permanent resident population'?

SELECT"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=500)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

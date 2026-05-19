"""
generate_dataset.py
-------------------
Generates a synthetic dataset of Indian social media tweets
about vaccines with sentiment labels.
Run this once to create dataset.csv
"""

import pandas as pd
import random

random.seed(42)

# ─── Positive tweets ──────────────────────────────────────────────
positive_tweets = [
    "Vaccines save lives and protect our families from deadly diseases",
    "Finally got my COVID vaccine today! Feeling proud and relieved",
    "Thanks to vaccination drive in India, polio is now eradicated",
    "Vaccine drives by Indian government are truly commendable",
    "Got vaccinated today at local PHC, process was smooth and quick",
    "Vaccination is the only way to achieve herd immunity in India",
    "My entire family is now fully vaccinated, feeling safe",
    "Covaxin made in India is a matter of national pride",
    "Free vaccination camps in rural areas is a great initiative",
    "Science has given us vaccines, let us use them wisely",
    "Took both doses of Covishield, side effects were mild and temporary",
    "Vaccine protects not just you but also elderly people around you",
    "India's vaccination rate has improved drastically this year",
    "Cowin portal made vaccine registration very easy and accessible",
    "Healthcare workers deserve respect for running vaccination camps",
    "Vaccines are backed by decades of scientific research and trials",
    "My grandmother got vaccinated at 80, she feels protected now",
    "Vaccine distribution in remote villages is really impressive",
    "Proud Indian, got my jab today! Jai Bharat",
    "The speed of India's vaccine rollout is amazing and inspiring",
    "Children are now safer thanks to childhood immunization programs",
    "BCG vaccine has saved millions of lives in India since decades",
    "Getting vaccinated is a social responsibility we all must fulfil",
    "Vaccine efficacy data clearly shows it reduces hospitalizations",
    "AIIMS doctors recommend vaccination for all eligible individuals",
    "Booster dose important for elderly and immunocompromised people",
    "Vaccination camps organized by RWA in our colony was very helpful",
    "Two doses done! I can now breathe easier in crowded places",
    "India achieved 1 billion vaccine doses milestone, historic moment",
    "Doctor explained vaccine benefits clearly, I am fully convinced",
    "Vaccines have helped India control second wave effectively",
    "School vaccination programs are essential for public health",
    "My parents trusted science and got vaccinated without hesitation",
    "Vaccine hesitancy is reducing gradually in Indian villages",
    "Community health workers doing great job promoting vaccination",
    "Pulse Polio campaign is a massive success story for India",
    "Got my flu vaccine, highly recommend everyone to do the same",
    "Vaccines are the greatest medical achievement of the 20th century",
    "India exports vaccines to other countries, we should be proud",
    "Serum Institute of India producing millions of doses is incredible",
    "Digital vaccine certificate from Cowin is very convenient",
    "Took vaccine, mild fever for one day, totally worth it for safety",
    "Vaccine education programs in schools should be encouraged more",
    "India's Jan Aushadhi and vaccination scheme helping poor families",
    "Got vaccinated at age 65, doctor said it is very important",
    "Vaccine rollout in tier 2 cities has been very organised",
    "Free vaccination for BPL families is an excellent government step",
    "Frontline workers vaccinated first was the right priority decision",
    "Vaccine creates antibodies that protect us from serious illness",
    "India's COVAXIN got WHO EUL approval, this is great news",
]

# ─── Negative tweets ──────────────────────────────────────────────
negative_tweets = [
    "I am scared of vaccine side effects, what if something goes wrong",
    "Vaccine queue at government hospital was chaotic and mismanaged",
    "Why is there no proper information about vaccine ingredients given",
    "Heard so many bad stories about vaccine reactions, I am worried",
    "The vaccine distribution is very unequal between rich and poor areas",
    "Cowin website keeps crashing, very frustrating experience",
    "Forced vaccination is against personal freedom and human rights",
    "Why should I trust a vaccine made in just one year of research",
    "My friend had severe allergic reaction after getting vaccinated",
    "Government data on vaccine side effects is not transparent enough",
    "Rural areas have no proper cold storage for vaccine distribution",
    "Vaccine shortage in our district is creating panic and frustration",
    "Why are private hospitals charging so much for vaccines",
    "Misinformation about vaccines is spreading faster than virus itself",
    "I developed fever and body pain after second dose, very unwell",
    "Vaccine efficacy drops over time, what is the point then",
    "Long waiting time at vaccination center is a big problem",
    "My elderly mother fainted at vaccination center due to heat",
    "Some batches of vaccines were found to be contaminated, scary",
    "No proper follow-up system after vaccination for monitoring",
    "Children vaccines causing autism is a genuine concern for parents",
    "Healthcare workers are exhausted and underpaid during campaigns",
    "Corporate companies profiting too much from pandemic vaccines",
    "Vaccine data from third world countries is not taken seriously",
    "Trust deficit in government health programs is a real problem",
    "Side effects from vaccine affected my work for three days",
    "Vaccine rollout was poorly planned in our municipality",
    "Why is there still no transparency about clinical trial data",
    "Online registration for vaccine is impossible for elderly people",
    "Heard that some vaccines cause blood clots, this is very dangerous",
    "Discrimination at vaccination center based on caste is shameful",
    "Fake vaccine certificates being sold openly is a serious issue",
    "Vaccine nationalism is preventing global herd immunity from forming",
    "My arm swelled badly after vaccination, doctor was not available",
    "Government is forcing vaccines without proper public consultation",
    "Quality control of vaccines in India needs much more scrutiny",
    "No compensation system for vaccine injury victims in India",
    "The whole vaccine drive feels like a political gimmick to me",
    "Rich countries hoarding vaccines while poor countries suffer",
    "Vaccine mandate is oppressive and violates individual rights",
    "I lost trust in vaccines after reading about Thalidomide scandal",
    "Why are some vaccine batches giving different reactions to people",
    "Healthcare infrastructure for vaccine rollout is completely inadequate",
    "Too much political interference in vaccine approval process",
    "People in my village refused vaccine due to wrong information",
    "Waste of vaccines due to poor storage management in many centers",
    "Why is booster dose needed so soon after full vaccination",
    "Vaccination data privacy concerns are completely ignored by authorities",
    "Black market for vaccines is proof the system is deeply broken",
    "No accountability when vaccine adverse events are reported",
]

# ─── Neutral tweets ───────────────────────────────────────────────
neutral_tweets = [
    "Vaccination drive started in our district today as per schedule",
    "Government announced new vaccine registration guidelines today",
    "CoWIN app updated with new features for vaccine certificate download",
    "Health ministry released new data on vaccine distribution numbers",
    "New vaccination center opened near railway station from Monday",
    "Vaccine eligibility now extended to 18 plus age group in India",
    "WHO released updated guidelines on COVID vaccine booster doses",
    "Government issued advisory on vaccine storage temperature requirements",
    "Third dose registration now open on Cowin portal for eligible",
    "Vaccine trial results to be published in medical journal next month",
    "Annual immunization schedule released by health department today",
    "Covaxin price fixed by government for private hospitals announced",
    "Serum Institute to produce additional doses for export programs",
    "Vaccination camp organized by local NGO in tribal belt area",
    "Mobile vaccination units deployed in urban slums this week",
    "Precautionary dose now available at government hospitals nearby",
    "Health worker training program for vaccine administration ongoing",
    "Vaccine certificate now linked with DigiLocker for easy access",
    "Survey on vaccine hesitancy conducted by university researchers",
    "Vaccination data being analyzed to understand coverage patterns",
    "New mRNA vaccine technology being studied by Indian researchers",
    "Vaccine cold chain management review conducted by ICMR team",
    "District health officer reviewed vaccination progress this week",
    "Vaccination numbers released by state government for last month",
    "NTAGI meeting held to discuss updated vaccine schedule protocol",
    "Vaccine adverse event reporting system updated with new features",
    "International health organizations monitoring India vaccine rollout",
    "Vaccine procurement process under review by parliamentary committee",
    "New vaccine for rotavirus included in national immunization program",
    "HPV vaccine rollout for adolescent girls discussed in health meet",
    "Vaccine awareness drive launched in schools by health department",
    "Doctors attending seminar on latest vaccine developments today",
    "State government compiled quarterly report on vaccination coverage",
    "Vaccine logistics management discussed in inter-ministerial meeting",
    "Nationwide survey on vaccine awareness to be conducted in January",
    "Research paper on vaccine effectiveness published by ICMR scientists",
    "Vaccine supply chain audit underway in northern states this month",
    "Health ministry clarified guidelines on vaccine mixing protocols",
    "Vaccination campaign coverage to be tracked using new digital tool",
    "Government hospitals to stock new batch of vaccines from next week",
    "Medical experts panel formed to study post-vaccination effects data",
    "Vaccine manufacturing capacity in India increased by thirty percent",
    "Vaccination records can now be downloaded via Aarogya Setu app",
    "Academic study on vaccine acceptance patterns among rural youth",
    "National immunization week observed across India with campaigns",
    "Health department organized orientation for new vaccination workers",
    "Parliamentary debate on national vaccine policy scheduled next week",
    "Government reviewing vaccine pricing structure for private sector",
    "Vaccine procurement committee meeting held by central government",
    "New guidelines on vaccine administration for pregnant women issued",
]

# ─── Build DataFrame ──────────────────────────────────────────────
tweets, labels = [], []

# We want ~1050 rows: 350 each
for t in positive_tweets:
    for _ in range(7):          # 50 × 7 = 350
        tweets.append(t)
        labels.append("positive")

for t in negative_tweets:
    for _ in range(7):
        tweets.append(t)
        labels.append("negative")

for t in neutral_tweets:
    for _ in range(7):
        tweets.append(t)
        labels.append("neutral")

# Shuffle
combined = list(zip(tweets, labels))
random.shuffle(combined)
tweets, labels = zip(*combined)

df = pd.DataFrame({"tweet": tweets, "label": labels})
df.to_csv("dataset.csv", index=False)
print(f"✅  Dataset generated: {len(df)} rows saved to dataset.csv")
print(df["label"].value_counts())

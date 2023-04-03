#! /usr/bin/env python3

import os
import openai
import sys
from datasets import load_dataset

dataset = load_dataset('boolq', split='validation[:30]')

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt_base="Answer the question given the passage: \nQ: is elder scrolls online the same as skyrim\nP: As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective.\nA: false\n\nQ: is batman and robin a sequel to batman forever\nP: With the box office success of Batman Forever in June 1995, Warner Bros. immediately commissioned a sequel. They hired director Joel Schumacher and writer Akiva Goldsman to reprise their duties the following August, and decided it was best to fast track production for a June 1997 target release date, which is a break from the usual 3-year gap between films. Schumacher wanted to homage both the broad camp style of the 1960s television series and the work of Dick Sprang. The storyline of Batman & Robin was conceived by Schumacher and Goldsman during pre-production on A Time to Kill. Portions of Mr. Freeze's back-story were based on the Batman: The Animated Series episode ``Heart of Ice'', written by Paul Dini.\nA: true\n\nQ: does a penalty shoot out goal count towards the golden boot\nP: A shoot-out is usually considered for statistical purposes to be separate from the match which preceded it. In the case of a two-legged fixture, the two matches are still considered either as two draws or as one win and one loss; in the case of a single match, it is still considered as a draw. This contrasts with a fixture won in extra time, where the score at the end of normal time is superseded. Converted shoot-out penalties are not considered as goals scored by a player for the purposes of their individual records, or for ``golden boot'' competitions.\nA: false\n\nQ: the boy in the plastic bubble based on true story\nP: The Boy in the Plastic Bubble is a 1976 American made-for-television drama film inspired by the lives of David Vetter and Ted DeVita, who lacked effective immune systems. It stars John Travolta, Glynnis O'Connor, Diana Hyland, Robert Reed, Ralph Bellamy & P.J. Soles. It was written by Douglas Day Stewart, executive produced by Aaron Spelling and Leonard Goldberg (who, at the time, produced Starsky and Hutch and Charlie's Angels), and directed by Randal Kleiser, who would work with Travolta again in Grease shortly after. The original music score was composed by Mark Snow. The theme song ``What Would They Say'' was written and sung by Paul Williams. William Howard Taft High School in Woodland Hills was used for filming.\nA: true\n\nQ: can someone die from a bullet shot in the air\nP: Bullets fired into the air usually fall back with terminal velocities much lower than their muzzle velocity when they leave the barrel of a firearm. Nevertheless, people can be injured, sometimes fatally, when bullets discharged into the air fall back down to the ground. Bullets fired at angles less than vertical are more dangerous as the bullet maintains its angular ballistic trajectory and is far less likely to engage in tumbling motion; it therefore travels at speeds much higher than a bullet in free fall.\nA: true\n\nQ: do blue and pink cotton candy taste the same\nP: Typically, once spun, cotton candy is only marketed by color. Absent a clear name other than 'blue', the distinctive taste of the blue raspberry flavor mix has gone on to become a compound flavor that some other foods (gum, ice cream, rock candy, fluoride toothpaste) occasionally borrow (``cotton-candy flavored ice cream'') to invoke the nostalgia of cotton candy that people typically only get to experience on vacation or holidays. Pink bubble gum went through a similar transition from specific branded product to a generic flavor that transcended the original confection, and 'bubble gum flavor' often shows up in the same product categories as 'cotton candy flavor'.\nA: false\n\nQ: is mount everest a part of the himalayas\nP: The Himalayan range has many of the Earth's highest peaks, including the highest, Mount Everest. The Himalayas include over fifty mountains exceeding 7,200 metres (23,600 ft) in elevation, including ten of the fourteen 8,000-metre peaks. By contrast, the highest peak outside Asia (Aconcagua, in the Andes) is 6,961 metres (22,838 ft) tall.\nA: true\n\nQ: can you get the death penalty as a minor\nP: Capital punishment for juveniles in the United States existed until March 1, 2005, when the U.S. Supreme Court banned it in Roper v. Simmons.\nA: false"

golds = []
preds = []

for item in dataset:
    golds.append(item['answer'])
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt_base + '\n\nQ: ' + item['question'] + '\nP: ' + item['passage'],
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    preds.append(response.choices[0]['text'])
correct = 0
with open("model_preds.txt", 'w') as f:
    f.write('pred|gold\n')
    for pred, gold in zip(preds, golds):
        if ('true' in pred) or ('false' in pred):
            pred_processed = ("true" in pred)
            correct += int(pred_processed == gold)
        else:
            print("poorly formatted response: " + pred + ", true answer: " + str(gold).lower(), file=sys.stderr)
        f.write(str('true' in pred).lower() + '|' + str(gold).lower() + '\n')
print(f'Davinci Accuracy: {correct} of {len(golds)} ({(correct / len(golds)) * 100}%)')

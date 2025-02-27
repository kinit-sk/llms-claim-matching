from typing import Any, List
import pandas as pd

class Prompt:
    """
    Prompt class to hold the prompt for the model
    """
    def __init__(self, **kwargs):
        self.prompt = None
        
        if 'template' in kwargs:
            self.from_template(kwargs['template'])

    def get_prompt(self, **kwargs) -> dict:
        return {
            'prompt': self.format(**kwargs)
        }
    
    def from_template(self, template: str, **kwargs):
        self.template = template
        
    def format(self, **kwargs) -> str:
        """
        Format the prompt using the saved prompt template.
        """
        return self.template.format(**kwargs)

    def __call__(self, **kwargs) -> Any:
        return self.get_prompt(
            **kwargs
        )

        
class PointwisePrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_prompt(self, **kwargs) -> dict:
        documents = kwargs.get('documents', [])
        query = kwargs.get('query', '')

        return {
            'prompt': [
                self.template.format(document=document, query=query)
                for document in documents
            ]
        }


class UniversalPrompt(Prompt):
    def __init__(self, **kwargs):
        pass

    def get_prompt(self, **kwargs) -> dict:
        documents = kwargs.get('documents', [])
        post = kwargs.get('query', '')
        
        docs = ' '.join(
            [f'{doc}' for doc in documents])

        return {
            'prompt': f'{docs}\n\n{post}'  # nopep8
        }
        
class AggregationPrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_prompt(self, **kwargs) -> dict:
        documents = kwargs.get('documents', [])
        query = kwargs.get('query', '')
        
        docs = '\n'.join(
            [f'Fact-checked claim {idx}: {doc}' for idx, doc in enumerate(documents)]
        )
        
        return {
            'prompt': self.template.format(document=docs, query=query)
        }

class FewShotPointwisePrompt(Prompt):
    def __init__(self, **kwargs):
        # multilingual examples
        self.english = kwargs.get('english', False)
        if 'examples_path' in kwargs:
            self.num_examples = kwargs.get('num_examples', 6)
            self.df = pd.read_csv(kwargs['examples_path'])
            self.df['random_examples'] = self.df['random_examples'].apply(eval)
        else:
            self.df = None
            self.examples = [
                ("People die after being vaccinated against COVID-19.", "This year the number of deaths increased rapidly due to COVID-19, but there is not any connection with the vaccines.", "Yes"),
                ("Soursop je zázračný prírodný zabijak rakovinových buniek a dokáže zabiť viac buniek ako chemoterapia.", "Kanabis preukázal potenciál byť zabijak rakoviny, pričom nový výskum naznačuje, že jeho zlúčeniny môžu mať protirakovinové vlastnosti, ktoré by mohli spôsobiť revolúciu v liečebných prístupoch.", "No"),
                ("Ryanair-Flugzeugabsturz in Indien während der Landung im Jahr 2023. 2 Piloten und 96 Passagiere tot, nur ein Passagier überlebte.", "Im Jahr 2023 kam es zu einem Flugzeugabsturz der Ryanair-Fluggesellschaft.", "Yes"),
                ("هبطت قوات المظليين العسكريين الروس في مدينة خاركوف بأوكرانيا.", "اختبارات PCR غير موثوقة وتعرض صحة الإنسان للخطر.", "No"),
                ("Το 1984, ο Ρόναλντ Ρίγκαν, ο τότε Πρόεδρος των ΗΠΑ, απαγόρευσε στους Νιγηριανούς να εισέλθουν στις ΗΠΑ μέσω του Μπουχάρι.", "Το 1984, κατά τη διάρκεια της προεδρίας του Ρόναλντ Ρίγκαν, οι Ηνωμένες Πολιτείες εφάρμοσαν ταξιδιωτικούς περιορισμούς στους Νιγηριανούς εν μέσω διπλωματικών εντάσεων γύρω από την ηγεσία του Νιγηριανού προέδρου Muhammadu Buhari." , "Yes"),
                ("포르투갈은 2001년에 모든 약물을 합법화했습니다. 포르투갈은 치료와 유해성 감소에 막대한 투자를 했습니다. 포르투갈은 현재 유럽에서 마약 관련 사망률이 가장 낮은 국가 중 하나입니다.", "아이슬란드에서는 약물 남용을 근절하기 위한 청소년 프로그램과 지역사회 참여에 중점을 두었습니다. 아이슬란드는 현재 유럽에서 청소년 약물 남용 비율이 가장 낮습니다.", "No"),
            ]
            self.num_examples = len(self.examples)
        self.prompt = None
        
        if 'template' in kwargs:
            self.from_template(kwargs['template'])

    def get_prompt(self, **kwargs) -> dict:
        documents = kwargs.get('documents', [])
        query = kwargs.get('query', '')
        
        post_text = 'post_text_en' if self.english else 'post_text'
        fact_check_text = 'factcheck_text_en' if self.english else 'factcheck_text'
        
        if self.df is not None:
            prompts = []
            for document in documents:
                examples = self.df[(self.df[post_text] == query) & (self.df[fact_check_text] == document)]['random_examples'].values[0]
                examples = examples[:self.num_examples]
                examples = [
                    self.template.format(document=doc, query=query).replace("\\n", "\n") + f'{label}'
                    for query, doc, label in examples
                ]
                
                text = '\n\n'.join(examples)
                prompts.append(f'{text}\n\n' + self.template.format(document=document, query=query).replace("\\n", "\n"))
            
            return {
                'prompt': prompts
            }
        else:
            # use template to create examples for few-shot learning
            examples = [
                self.template.format(document=doc, query=query).replace("\\n", "\n") + f'{label}'
                for doc, query, label in self.examples
            ]
            
            # create text from examples
            text = '\n\n'.join(examples)
            return {
                'prompt': [
                    f'{text}\n\n' + self.template.format(document=document, query=query).replace("\\n", "\n")
                    for document in documents
                ]
            }
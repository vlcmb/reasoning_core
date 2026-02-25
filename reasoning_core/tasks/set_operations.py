import random
import numpy as np
import datetime
from num2words import num2words
from dataclasses import dataclass
from reasoning_core.template import Task, Problem, Config
import itertools
import string
from ast import literal_eval


### Tool functions 🛠️

class SetList(list):
    def __repr__(self):
        # f-strings call __str__, which for lists defaults to calling this.
        return "{" + ", ".join(map(repr, self)) + "}"

def return_shuffle(domain):
    """Domain must be a collection of element convertible in list"""
    # Cast to SetList so it prints with curly braces but behaves like a list
    domain = SetList(domain)
    random.shuffle(domain)
    return domain

def random_subdomain(domain, size=None):
    """Domain must be a collection of element convertible in list, and frac must be a float between 0 and 1. \n
    return a random fraction of the domain."""
    domain = list(domain)
    subset = random.sample(domain, size)
    return return_shuffle(subset)

def create_intension(domain : list, length : int):
        """Returns a contiguous subdomain (of domain) of size length."""
        n = len(domain)
        i = np.random.randint(n-length)
        return domain[i:i+length]

def make_domains(size):
    
    NUM = [int(i) for i in range(1,size+1)]
    NUM_EN = [num2words(i,lang='en') for i in NUM]
    NUM_FR = [num2words(i,lang='fr') for i in NUM]
    
    start = (datetime.date(2020, 1, 1))
    DATES = [(start + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(size)]
    DATES_EN = [(start + datetime.timedelta(days=i)).strftime('%B %d, %Y') for i in range(size)]
    
    gen = itertools.chain.from_iterable((''.join(p) for p in itertools.product(string.ascii_lowercase, repeat=n)) for n in itertools.count(1))
    LETTERS = list(itertools.islice(gen, size))

    domains = [NUM, NUM_EN, DATES, DATES_EN, LETTERS]
    return domains

def perturb_list(input_l, base_domain, n_perturbation=1):
    for _ in range(n_perturbation):
        perturbation = random.choice(['add', 'remove', 'replace'])
        lst = input_l[:]
        complementary = list(set(base_domain) - set(lst))
        if perturbation == 'add':
            new_element = random.choice(complementary)
            insert_pos = random.randint(0, len(lst))
            lst.insert(insert_pos, new_element)
        elif perturbation == 'remove' and len(lst) > 1:
            del lst[random.randint(0, len(lst) - 1)]
        elif perturbation == 'replace':
            index = random.randint(0, len(lst) - 1)
            lst[index] = random.choice(complementary)
    return (lst , perturbation)


def intersection_metric(set1, set2):
    return len(set1 & set2)/len(set1 | set2)
    
### Task class 🎮 🎯

@dataclass
class SetOpsConfig(Config):
    """
    Configuration for core Set Theory logical evaluation tasks.
    
    | Parameter | Type | Default | Utility |
    | :--- | :--- | :--- | :--- |
    | `domain_size` | `int` | `1000` | Number of distinct elements available within the generated universe domain pool. |
    | `set_size` | `int` | `8` | Target baseline subset length drawn from the base universe. |
    | `n_max_perturbation` | `int` | `2` | Maximum structural edits applied to a subset to create dynamic sequence variations. |
    | `prob_equal` | `float` | `0.5` | Probability threshold for intentionally generating structurally or semantically equal sets. |
    | `n_domains` | `int` | `1` | Number of distinct thematic entity domains (numbers, names, dates, letters) utilized. |
    """
    domain_size: int = 1000
    set_size: int = 8
    n_max_perturbation: int = 2
    prob_equal: float = 0.5
    n_domains : int = 1
    def update(self, c):
        self.set_size *= 1 + c
        self.domain_size *= 1 + c
        self.n_max_perturbation *= 1 + c
        self.n_domains += c
        
class SetIntersection(Task):
    """
    Task responsible for calculating discrete subset intersections.
    
    The model receives two implicitly rendered string arrays (`Set1`, `Set2`). It must logically formulate and return pythonic notation ` {element1, element2} ` representing the exact intersection overlap values.
    """
    def __init__(self, config=SetOpsConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)
    
    def generate(self):
        chosen_domain = random.choice(self.domains[:self.config.n_domains])
        N=6
        set_1 = random_subdomain( chosen_domain, size=self.config.set_size) 
        others = list(set(chosen_domain) - set(set_1))
        set_2 = random.sample(others, N//2) + random.sample(set_1, N//2)

        inter = sorted(list(set(set_1) & set(set_2)))
        inter = "{" + ", ".join(map(repr, inter)) + "}"

        meta = {}
        meta["set_1"] = return_shuffle(set_1)
        meta["set_2"] = return_shuffle(set_2)
        return Problem(metadata = meta, answer = inter)
     
    def prompt(self, metadata) -> str:
        return (
            f"Set1: {metadata['set_1']}\n"
            f"Set2: {metadata['set_2']}\n"
            "Only return the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}."
        )

    def score_answer(self, answer, entry):
        reference = entry['answer']
        try:
            set_pred , set_truth  = set(literal_eval(answer)),set(literal_eval(reference))
            if set_truth == set():
                return int(set_pred == set())
    
            return intersection_metric(set_pred, set_truth)
        except:
            return 0

@dataclass
class SetMissingElementConfig(SetOpsConfig):
    """
    Configuration for subset deduction reasoning evaluation.
    
    | Parameter | Type | Default | Utility |
    | :--- | :--- | :--- | :--- |
    | `set_size` | `int` | `10` | Specific subset length explicitly overridden for missing element induction reasoning tasks. |
    | `prob_no_missing` | `float` | `0.1` | Statistical likelihood that the generated test set variation intentionally drops exactly zero underlying elements. |
    """
    set_size: int = 10
    prob_no_missing: float = 0.1
    def update(self, c):
        self.set_size *= 1 + c
        self.n_max_perturbation *= 1 + c
        self.domain_size *= 1 + c
        self.n_domains += c

class SetMissingElement(Task):
    """
    Task responsible for complementary subset identification.
    
    The model is provided an implicitly ordered sub-sequence spanning a distinct semantic context (like alphabetical dates or numbers). It must logically deduce elements arbitrarily missing/stripped from the underlying sequence format.
    """
    def __init__(self, config=SetMissingElementConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)
        
    def generate(self):
        chosen_domain = random.choice(self.domains[:self.config.n_domains])
        intention = create_intension(chosen_domain, self.config.set_size)
        n_missing = 0 if random.random() < self.config.prob_no_missing else random.randint(1, 3)
        removable = intention[1:-1]
        missing = sorted(random.sample(removable, min(n_missing, len(removable))), key=str)
        for e in missing: intention.remove(e)
        answer = "{" + ", ".join(map(repr, missing)) + "}"
        return Problem(metadata={'element_list': return_shuffle(intention)}, answer=answer)

    def prompt(self, metadata) -> str:
        return (
            f"Set_A: {metadata['element_list']}\n"
            "Only return the missing elements from Set_A as a Python set."
        )

    def score_answer(self, answer, entry):
        try:
            pred, truth = set(literal_eval(answer)), set(literal_eval(entry['answer']))
            return int(pred == truth) if not truth else intersection_metric(pred, truth)
        except:
            return 0

@dataclass
class CountElementsConfig(Config):
    """
    Configuration for contextual list frequency evaluation.
    
    | Parameter | Type | Default | Utility |
    | :--- | :--- | :--- | :--- |
    | `max_count` | `int` | `3` | Upper limit bounding the frequency occurrence of a single target element within a populated array. |
    | `list_size` | `int` | `10` | The constrained length of the overall target discrete numerical/string array. |
    | `domain_size` | `int` | `20` | Distinct items drawn upon to procedurally populate the random remainder of the target array logic. |
    """
    max_count: int = 3
    list_size: int = 10
    domain_size: int = 20
    def update(self, c):
        self.max_count += c
        self.list_size += c
        self.domain_size *= 1 + c

class CountElements(Task):
    """
    Task responsible for discrete list element aggregation scoring.
    
    The model receives a flat sequence iteration and an explicit target subset query. It must mathematically output the correct integer occurrence count enumerating the number of times the target sequence strictly appears in the list.
    """
    def __init__(self, config=CountElementsConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)

    def generate(self):
        count = random.randint(0, self.config.max_count)
        domain = random.choice(self.domains)
        target = random.choice(domain)
        others = [e for e in domain if e != target]
        n_others = self.config.list_size - count
        elements = [target] * count + random.choices(others, k=n_others)
        random.shuffle(elements)
        return Problem(metadata={'elements': elements, 'target': target}, answer=str(count))

    def prompt(self, metadata) -> str:
        return f"List: {metadata['elements']}\nHow many times does {metadata['target']!r} appear? Only return the number."

    def score_answer(self, answer, entry):
        try: return 1 / (1 + abs(int(answer.strip()) - int(entry['answer'])))
        except: return 0

@dataclass
class SetEquality(Task):
    """
    Task responsible for Boolean bipartite sequence equivalence deduction.
    
    The model receives two discretely sampled random sub-domains. It must mathematically and structurally deduce whether the two implicit configurations are strictly equivalent, outputting True or False.
    """
    def __init__(self, config=SetOpsConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)

    def generate(self):
        chosen_domain =random.choice(self.domains)
        subset = random_subdomain(chosen_domain, size = self.config.set_size)
        meta = {}
        meta["base_subset"] = return_shuffle(subset)
        subset_bis = subset
        perturbation = None
        n_pert = random.choice([k+1 for k in range(self.config.n_max_perturbation)])
        if random.random() > self.config.prob_equal:
            subset_bis , perturbation = perturb_list(subset, chosen_domain, n_pert)
        meta["subset_bis"] = return_shuffle(subset_bis) 
        meta["perturbation"] = perturbation
        return Problem(metadata = meta, answer = str((set(subset) == set(subset_bis))))

    def prompt(self, metadata) -> str:
        return (
            f"Set1: {metadata['base_subset']}\n"
            f"Set2: {metadata['subset_bis']}\n"
            "Only return True if Set1 and Set2 contain exactly the same elements, False otherwise."
        )
    


from arguments import get_args_parser
import random
import json

mapping = {"per": "person", "org": "organization"}


def get_temps_re_wiki80():
    args = get_args_parser()
    temps = {}
    with open(args.rel2id_dir, "r") as f:
        rel2id = json.loads(f.read())

    for name, id in rel2id.items():
        labels = name.split(' ')
        temps[name] = labels

    return temps


def get_temps_re():
    args = get_args_parser()
    temps = {}
    with open(args.rel2id_dir, "r") as f:
        rel2id = json.loads(f.read())

    for name, id in rel2id.items():
        labels = name.split(':')
        if len(labels) == 2:
            if labels[0] in mapping.keys():
                label0 = mapping[labels[0]]
            else:
                raise NotImplementedError("Relation {} is not in mapping dict.".format(labels[0]))
            label1 = labels[1].split("_")
            label1_ = []
            for index, l in enumerate(label1):
                if "/" in l:
                    l = l.replace("/", "or")
                if "or" in l and not l.startswith("or"):
                    index_or = l.index("or")
                    l_list = [l[:index_or], l[index_or: index_or + 2], l[index_or + 2:]]
                    label1_ += l_list
                else:
                    label1_ += [l]
        elif labels == ["NA"] or ["no_relation"]:
            label0 = "entity"
            label1_ = ["no", "relation",]

        labels_tokens = [label0] + label1_
        temps[name] = labels_tokens

    return temps


# for TACRED
valid_conditions_tacred = {'org:founded_by': ['organization:person'],
                    'no_relation': ['person:misc', 'person:nationality', 'organization:number', 'organization:country', 'person:location', 'organization:date', 'person:cause_of_death', 'person:country', 'person:number', 'person:criminal_charge', 'organization:organization', 'person:title', 'organization:religion', 'person:date', 'person:person', 'organization:person', 'person:state_or_province', 'organization:misc', 'organization:url', 'organization:city', 'person:religion', 'person:organization', 'person:duration', 'organization:location', 'organization:ideology', 'person:city', 'organization:state_or_province'],
                    'per:employee_of': ['person:organization', 'person:location'],
                    'org:alternate_names': ['organization:organization', 'organization:misc'],
                    'per:cities_of_residence': ['person:city', 'person:location'],
                    'per:children': ['person:person'],
                    'per:title': ['person:title'],
                    'per:siblings': ['person:person'],
                    'per:religion': ['person:religion'],
                    'per:age': ['person:duration', 'person:number'],
                    'org:website': ['organization:url'],
                    'per:stateorprovinces_of_residence': ['person:state_or_province', 'person:location'],
                    'org:member_of': ['organization:location', 'organization:country', 'organization:state_or_province', 'organization:organization'],
                    'org:top_members/employees': ['organization:person'],
                    'per:countries_of_residence': ['person:country', 'person:nationality', 'person:location'],
                    'org:city_of_headquarters': ['organization:location', 'organization:city'],
                    'org:members': ['organization:country', 'organization:organization'],
                    'org:country_of_headquarters': ['organization:location', 'organization:country'],
                    'per:spouse': ['person:person'],
                    'org:stateorprovince_of_headquarters': ['organization:location', 'organization:state_or_province'],
                    'org:number_of_employees/members': ['organization:number'],
                    'org:parents': ['organization:location', 'organization:country', 'organization:state_or_province', 'organization:organization'],
                    'org:subsidiaries': ['organization:location', 'organization:country', 'organization:organization'],
                    'per:origin': ['person:country', 'person:nationality', 'person:location'],
                    'org:political/religious_affiliation': ['organization:ideology', 'organization:religion'],
                    'per:other_family': ['person:person'],
                    'per:stateorprovince_of_birth': ['person:state_or_province'],
                    'org:dissolved': ['organization:date'],
                    'per:date_of_death': ['person:date'],
                    'org:shareholders': ['organization:organization', 'organization:person'],
                    'per:alternate_names': ['person:misc', 'person:person'],
                    'per:parents': ['person:person'],
                    'per:schools_attended': ['person:organization'],
                    'per:cause_of_death': ['person:cause_of_death'],
                    'per:city_of_death': ['person:city', 'person:location'],
                    'per:stateorprovince_of_death': ['person:state_or_province', 'person:location'],
                    'org:founded': ['organization:date'],
                    'per:country_of_birth': ['person:country', 'person:nationality', 'person:location'],
                    'per:date_of_birth': ['person:date'],
                    'per:city_of_birth': ['person:city', 'person:location'],
                    'per:charges': ['person:criminal_charge'],
                    'per:country_of_death': ['person:country', 'person:nationality', 'person:location']}

# for TACREV
valid_conditions_tacrev = {'org:founded_by': ['organization:person'],
                    'no_relation': ['person:city', 'organization:location', 'person:country', 'organization:state_or_province', 'organization:person', 'person:organization', 'person:location', 'organization:ideology', 'person:criminal_charge', 'organization:misc', 'person:religion', 'organization:date', 'person:state_or_province', 'person:date', 'person:duration', 'person:number', 'organization:number', 'person:title', 'organization:religion', 'person:cause_of_death', 'organization:country', 'person:misc', 'organization:url', 'organization:organization', 'person:person', 'organization:city', 'person:nationality'],
                    'per:employee_of': ['person:city', 'person:location', 'person:country', 'person:title', 'person:person', 'person:organization', 'person:state_or_province'],
                    'org:alternate_names': ['organization:organization', 'organization:misc'],
                    'per:cities_of_residence': ['person:city', 'person:location', 'person:country', 'person:person', 'person:organization'],
                    'per:children': ['person:city', 'person:person'],
                    'per:title': ['person:organization', 'person:title', 'person:person'],
                    'per:siblings': ['person:person'],
                    'per:religion': ['person:organization', 'person:religion'],
                    'per:age': ['person:duration', 'person:number'],
                    'org:website': ['organization:organization', 'organization:url'],
                    'per:stateorprovinces_of_residence': ['person:location', 'person:state_or_province'],
                    'org:member_of': ['organization:country', 'organization:organization', 'organization:location', 'organization:state_or_province'],
                    'org:top_members/employees': ['organization:organization', 'organization:person'],
                    'per:countries_of_residence': ['person:city', 'person:location', 'person:nationality', 'person:country'],
                    'org:city_of_headquarters': ['organization:location', 'organization:state_or_province', 'organization:person', 'organization:city', 'organization:misc'],
                    'org:members': ['organization:country', 'organization:organization'],
                    'org:country_of_headquarters': ['organization:country', 'organization:location', 'organization:organization', 'organization:misc'],
                    'per:spouse': ['person:title', 'person:person'],
                    'org:stateorprovince_of_headquarters': ['organization:location', 'organization:state_or_province', 'organization:misc'],
                    'org:number_of_employees/members': ['organization:number'],
                    'org:parents': ['organization:country', 'organization:location', 'organization:state_or_province', 'organization:organization', 'organization:person'],
                    'org:subsidiaries': ['organization:country', 'organization:location', 'organization:organization', 'organization:person', 'organization:misc'],
                    'per:origin': ['person:location', 'person:nationality', 'person:country'],
                    'org:political/religious_affiliation': ['organization:religion', 'organization:ideology'],
                    'per:other_family': ['person:person'],
                    'per:stateorprovince_of_birth': ['person:city', 'person:state_or_province'],
                    'org:dissolved': ['organization:date'],
                    'per:date_of_death': ['person:date'],
                    'org:shareholders': ['organization:organization', 'organization:person'],
                    'per:alternate_names': ['person:organization', 'person:misc', 'person:person'],
                    'per:parents': ['person:person'],
                    'per:schools_attended': ['person:organization', 'person:city', 'person:location'],
                    'per:cause_of_death': ['person:criminal_charge', 'person:cause_of_death'],
                    'per:city_of_death': ['person:city', 'person:location', 'person:state_or_province'],
                    'per:stateorprovince_of_death': ['person:location', 'person:state_or_province', 'person:nationality'],
                    'org:founded': ['organization:date'],
                    'per:country_of_birth': ['person:location', 'person:nationality', 'person:country'],
                    'per:date_of_birth': ['person:date'],
                    'per:city_of_birth': ['person:city', 'person:location', 'person:state_or_province'],
                    'per:charges': ['person:criminal_charge'],
                    'per:country_of_death': ['person:location', 'person:nationality', 'person:country']}

# for RE-TACRED
valid_conditions_re_tacred = {'org:founded_by': ['organization:person'],
                    'no_relation': ['organization:state_or_province', 'organization:date', 'organization:organization', 'person:state_or_province', 'organization:religion', 'person:title', 'person:cause_of_death', 'organization:ideology', 'person:date', 'organization:city', 'person:duration', 'person:organization', 'organization:location', 'person:city', 'organization:person', 'person:nationality', 'person:person', 'person:criminal_charge', 'person:country', 'organization:number', 'organization:url', 'person:number', 'organization:country', 'person:religion', 'person:location'],
                    'per:identity': ['person:person'],
                    'org:alternate_names': ['organization:organization'],
                    'per:children': ['person:person'],
                    'per:origin': ['person:country', 'person:city', 'person:nationality', 'person:location'],
                    'per:countries_of_residence': ['person:city', 'person:nationality', 'person:country', 'person:location'],
                    'per:employee_of': ['person:country', 'person:organization', 'person:city', 'person:nationality', 'person:state_or_province', 'person:location'],
                    'per:title': ['person:religion', 'person:title'],
                    'org:city_of_branch': ['organization:state_or_province', 'organization:location', 'organization:country', 'organization:city'],
                    'per:religion': ['person:religion', 'person:title'],
                    'per:age': ['person:date', 'person:number', 'person:duration', 'person:title'],
                    'per:date_of_death': ['person:date', 'person:number', 'person:duration'],
                    'org:website': ['organization:url', 'organization:date', 'organization:number'],
                    'per:stateorprovinces_of_residence': ['person:city', 'person:country', 'person:state_or_province', 'person:location'],
                    'org:top_members/employees': ['organization:person'],
                    'org:number_of_employees/members': ['organization:number'],
                    'org:members': ['organization:city', 'organization:state_or_province', 'organization:location', 'organization:organization', 'organization:country'],
                    'org:country_of_branch': ['organization:state_or_province', 'organization:location', 'organization:country', 'organization:city'],
                    'per:spouse': ['person:person'],
                    'org:stateorprovince_of_branch': ['organization:state_or_province', 'organization:location', 'organization:country', 'organization:city'],
                    'org:political/religious_affiliation': ['organization:religion', 'organization:ideology'],
                    'org:member_of': ['organization:organization'],
                    'per:siblings': ['person:person'],
                    'per:stateorprovince_of_birth': ['person:city', 'person:nationality', 'person:state_or_province', 'person:location'],
                    'org:dissolved': ['organization:date'],
                    'per:other_family': ['person:person'],
                    'org:shareholders': ['organization:person', 'organization:organization'],
                    'per:parents': ['person:person'],
                    'per:charges': ['person:cause_of_death', 'person:criminal_charge', 'person:title'],
                    'per:schools_attended': ['person:organization'],
                    'per:cause_of_death': ['person:cause_of_death', 'person:date', 'person:criminal_charge', 'person:title'],
                    'per:city_of_death': ['person:city', 'person:country', 'person:state_or_province', 'person:location'],
                    'per:stateorprovince_of_death': ['person:city', 'person:nationality', 'person:state_or_province', 'person:location'],
                    'org:founded': ['organization:date'],
                    'per:country_of_death': ['person:country', 'person:city', 'person:nationality'],
                    'per:country_of_birth': ['person:nationality', 'person:country', 'person:state_or_province'],
                    'per:date_of_birth': ['person:date', 'person:number', 'person:duration'],
                    'per:cities_of_residence': ['person:city', 'person:country', 'person:state_or_province', 'person:location'],
                    'per:city_of_birth': ['person:city', 'person:nationality', 'person:state_or_province', 'person:location']}

valid_conditions_all = {"tacred": valid_conditions_tacred, "tacrev": valid_conditions_tacrev, "re-tacred": valid_conditions_re_tacred}


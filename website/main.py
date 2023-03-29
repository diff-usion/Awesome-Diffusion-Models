import itertools

from jinja2 import Template
import json


DOC_DIR = '../docs'

class Link:
    def __init__(self, name, href):
        self.name = name
        self.href = href


class Paper:
    def __init__(self, data):
        self.title = data['title']
        self.authors = data['authors']
        self.source = data.get('source', None)
        self.date = data.get('date', None)
        self.links = [Link(k, v) for k, v in data['links'].items()]
        self.field = data['field']
        self.task = data['task']


class Field:
    def __init__(self, name, tasks):
        self.name = name
        self.tasks = [Task(k, v, self) for k, v in tasks.items()]

    @property
    def total(self):
        return sum([len(task.papers) for task in self.tasks])

    @property
    def url_name(self):
        return self.name.lower().replace(' ', '_')

    @property
    def papers(self):
        papers = list(itertools.chain(*[task.papers for task in self.tasks]))
        papers = sorted(papers, key=lambda x: x.date)
        papers.reverse()
        return papers


class Task:
    def __init__(self, name, papers, field):
        self.name = name
        self._papers = papers
        self.field = field

    @property
    def url_name(self):
        return self.field.url_name + '_' + self.name.lower().replace(' ', '_')

    @property
    def papers(self):
        papers = self._papers
        try:
            papers = sorted(papers, key=lambda x: x.date)
            papers.reverse()
        except:
            pass
        return papers


def find_all_fields(papers):
    fields = {}
    for paper in papers:
        if paper.field not in fields:
            fields[paper.field] = {}
        if paper.task not in fields[paper.field]:
            fields[paper.field][paper.task] = set()
        fields[paper.field][paper.task].add(paper)

    field_objs = []
    for k, v in fields.items():
        obj = Field(k, v)
        field_objs.append(obj)

    return field_objs


def build_paper():
    with open('db.json', 'r') as f:
        db = json.load(f)

    papers = [Paper(paper) for paper in db['papers']]

    fields = find_all_fields(papers)

    with open(f'{DOC_DIR}/template.html', 'r') as fin:
        template = Template(fin.read(), lstrip_blocks=True, trim_blocks=True)

        first = True
        for field in fields:
            for task in field.tasks:
                out = template.render(fields=fields, papers=task.papers, highlight_task=task, section='paper')
                with open(f'{DOC_DIR}/{task.url_name}.html', 'w', encoding='utf-8') as fout:
                    fout.write(out)


def build_field():
    with open('db.json', 'r') as f:
        db = json.load(f)

    papers = [Paper(paper) for paper in db['papers']]

    fields = find_all_fields(papers)

    with open(f'{DOC_DIR}/template.html', 'r') as fin:
        template = Template(fin.read(), lstrip_blocks=True, trim_blocks=True)

        first = True
        for field in fields:
            out = template.render(fields=fields, papers=field.papers, section='paper')
            with open(f'{DOC_DIR}/{field.url_name}.html', 'w', encoding='utf-8') as fout:
                fout.write(out)
            if first:
                with open(f'{DOC_DIR}/index.html', 'w', encoding='utf-8') as fout:
                    fout.write(out)
                first = False


def build_resource():
    with open('resource.json', 'r') as f:
        db = json.load(f)

    papers = [Paper(paper) for paper in db['resources']]

    fields = find_all_fields(papers)

    with open(f'{DOC_DIR}/template.html', 'r') as fin:
        template = Template(fin.read(), lstrip_blocks=True, trim_blocks=True)

        first = True
        for field in fields:
            for task in field.tasks:
                out = template.render(fields=fields, papers=task.papers, highlight_task=task, section='resource')
                with open(f'{DOC_DIR}/{task.url_name}.html', 'w') as fout:
                    fout.write(out)
                if first:
                    with open(f'{DOC_DIR}/resource.html', 'w') as fout:
                        fout.write(out)
                    first = False


if __name__ == '__main__':
    build_paper()
    build_resource()
    build_field()

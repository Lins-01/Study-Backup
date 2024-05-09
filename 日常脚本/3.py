import re

# 函数：将完整名转换为首字母缩写
def abbreviate_name(name):
    parts = name.strip().split(', ')
    return f"{parts[1][0]}. {parts[0]}" if len(parts) == 2 else name

# 函数：转换引用格式
def convert_references(refs):
    # 先捕获所有字段，然后再提取，这样可以不用管字段的出现顺序
    pattern = re.compile(
        r'@(\w+){([^,]+),'
        r'((?:\s*\w+\s*=\s*{[^}]+},)*)'
        r'\s*}', re.DOTALL
    )

    converted_refs = []
    for ref in re.finditer(pattern, refs):
        entry_type, identifier, fields_str = ref.groups()
        fields = re.findall(r'\s*(\w+)\s*=\s*{([^}]+)}', fields_str)
        field_dict = {field[0].lower(): field[1] for field in fields}

        author_str = ''
        if 'author' in field_dict:
            author_list = field_dict['author'].split(' and ')
            first_three_authors = [abbreviate_name(author) for author in author_list[:3]]
            author_str = ', '.join(first_three_authors) + (' et al.' if len(author_list) > 3 else '')

        title = field_dict.get('title', '')
        journal = field_dict.get('journal', '')
        volume = field_dict.get('volume', '')
        number = field_dict.get('number', '')
        pages = field_dict.get('pages', '')
        year = field_dict.get('year', '')
        doi = field_dict.get('doi', '')

        converted = f"\\bibitem{{{identifier}}} {author_str}, {title}. {journal} {{\\bf {volume}}}, {pages} ({year}).   \\href{{https://doi.org/{doi}}}{{doi: {doi}}}"
        converted_refs.append(converted)
    return converted_refs


references = """
@article{ref1,
  title={Renewable energy and nuclear power towards sustainable development: Characteristics and prospects},
  author={ Karakosta, Charikleia  and  Pappas, Charalampos  and  Marinakis, Vangelis  and  Psarras, John },
  journal={Renewable and sustainable energy reviews},
  volume={22},
  pages={187-197},
  year={2013},
}

@article{ref2,
  title={Modeling and control of nuclear reactor cores for electricity generation: A review of advanced technologies},
  author={ Li, Gang  and  Wang, Xueqian  and  Liang, Bin  and  Li, Xiu  and  Zhang, Bo  and  Zou, Yu },
  journal={Renewable and Sustainable Energy Reviews},
  volume={60},
  pages={116-128},
  year={2016},
}

@article{ref3,
title = {Modelling of Large Nuclear Reactors to Control Power Density Distribution},
journal = {IFAC Proceedings Volumes},
volume = {23},
number = {8, Part 6},
pages = {85 - 90},
year = {1990},
author = {J. Kalkkuhl and B. Schmidt and W. Schröder and R. Strietzel},
}
"""
# # 调用转换函数并打印结果
# converted_references = convert_references(references)
# for ref in converted_references:
#     print(ref)


# 主逻辑
# 替换以下路径为您的ref.bib文件路径
file_path = 'ref.bib'
# 读取BibTeX文件
def read_bibtex_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
bib_content = read_bibtex_file(file_path)
converted_references = convert_references(bib_content)

# 将转换后的引用保存到一个新文件
output_file_path = '2converted_references.txt'
with open(output_file_path, 'w') as file:
    for ref in converted_references:
        file.write(ref + "\n"+"\n")

print(f"Converted references saved to {output_file_path}")
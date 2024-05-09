import re

# 更新正则表达式以匹配不同类型的条目
pattern = re.compile(r'@(article|inproceedings|book){([^,]+),'
                     r'\s*title={([^}]+)},'
                     r'\s*author={([^}]+)},'
                     r'\s*journal={([^}]+)*},?'
                     r'\s*volume={([^}]+)*},?'
                     r'\s*pages={([^}]+)*},?'
                     r'\s*year={([^}]+)},'
                     r'(?:\s*number=\s*{([^}]+)},)?'
                     r'\s*}', re.DOTALL)

# 函数：将完整名转换为首字母缩写
def abbreviate_name(name):
    parts = name.strip().split(', ')
    return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else name

# 函数：转换引用格式
def convert_references(refs):
    converted_refs = []
    for ref in re.finditer(pattern, refs):
        entry_type, identifier, title, authors, journal, volume, pages, year, number = ref.groups()
        if authors:
            author_list = authors.split(' and ')
            first_three_authors = [abbreviate_name(author) for author in author_list[:3]]
            author_str = ', '.join(first_three_authors) + (' et al.' if len(author_list) > 3 else '')
        else:
            author_str = ''
        journal_str = f'. {journal}' if journal else ''
        volume_str = f' {{\\bf {volume}}}' if volume else ''
        pages_str = f', {pages}' if pages else ''
        converted = f"\\bibitem{{{identifier}}} {author_str}, {title}{journal_str}{volume_str}{pages_str} ({year}).   \\href{{https://doi.org/placeholder}}{{doi: placeholder}}"
        converted_refs.append(converted)
    return converted_refs

# 示例参考文献字符串（需要替换为您自己的参考文献内容）
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

# 调用转换函数并打印结果
converted_references = convert_references(references)
for ref in converted_references:
    print(ref)

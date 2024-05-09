import re
import json

# 读取BibTeX文件
def read_bibtex_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()



# 函数：转换引用格式
def convert_references(refs):
    # 正则表达式模式，匹配包含可选字段的参考文献条目
    # pattern = re.compile(
    #     r'@(\w+){([^,]+),'
    #     r'(?:\s*title\s*=\s*{([^}]+)},)?'
    #     r'(?:\s*journal\s*=\s*{([^}]+)},)?'
    #     r'(?:\s*volume\s*=\s*{([^}]+)},)?'
    #     r'(?:\s*number\s*=\s*{([^}]+)},)?'
    #     r'(?:\s*pages\s*=\s*{([^}]+)},)?'
    #     r'(?:\s*year\s*=\s*{([^}]+)},)?'
    #     r'(?:\s*author\s*=\s*{([^}]+)},)?'
    #     r'\s*}', re.DOTALL
    # )
    pattern = re.compile(
    r'@(\w+){([^,]+),'
    r'\s*title={([^}]+)},'
    r'(?:\s*author={([^}]+)},)?'
    r'(?:\s*journal={([^}]+)},)?'
    r'(?:\s*volume={([^}]+)},)?'
    r'(?:\s*number\s*=\s*{([^}]+)},)?'
    r'(?:\s*month={([^}]+)},)?'
    r'(?:\s*publisher={([^}]+)},)?'
    r'(?:\s*bookTitle={([^}]+)},)?'
    r'(?:\s*address={([^}]+)},)?'
    r'(?:\s*pages={([^}]+)},)?'
    r'\s*year={([^}]+)},'
    r'\s*}', re.DOTALL
)


    # 函数：将完整名转换为首字母缩写
    def abbreviate_name(name):
        parts = name.strip().split(', ')
        return f"{parts[1][0]}. {parts[0]}" if len(parts) == 2 else name
    

    converted_refs = []
    for ref in re.finditer(pattern, refs):
        entry_type, identifier, title, journal, volume, number, pages, year, author, month, publisher, bookTitle, address   = ref.groups()
        print(author)
        author_str = ''
        if author:
            author_list = author.split(' and ')
            first_three_authors = [abbreviate_name(author) for author in author_list[:3]]
            author_str = ', '.join(first_three_authors) + (' et al.' if len(author_list) > 3 else '')
        converted = f"\\bibitem{{{identifier}}} {author_str}, {title}. {journal} {{\\bf {volume}}}, {pages} ({year}).   \\href{{https://doi.org/placeholder}}{{doi: placeholder}}"
        converted_refs.append(converted)
    return converted_refs

# 示例参考文献字符串（假设这是从文件中读取的内容）
# references = """
# @article{ref3,
# title = {Modelling of Large Nuclear Reactors to Control Power Density Distribution},
# journal = {IFAC Proceedings Volumes},
# volume = {23},
# number = {8, Part 6},
# pages = {85 - 90},
# year = {1990},
# author = {J. Kalkkuhl and B. Schmidt and W. Schröder and R. Strietzel},
# }
# """

# 主逻辑
# 替换以下路径为您的ref.bib文件路径
file_path = 'ref.bib'
bib_content = read_bibtex_file(file_path)
converted_references = convert_references(bib_content)

# 将转换后的引用保存到一个新文件
output_file_path = 'converted_references.txt'
with open(output_file_path, 'w') as file:
    for ref in converted_references:
        file.write(ref + "\n")

print(f"Converted references saved to {output_file_path}")

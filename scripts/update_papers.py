import arxiv
import json
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict

class PaperUpdater:
    def __init__(self,paper_path):
        self.existing_papers = set()
        self.keywords = self.load_keywords()
        self.paper_path = paper_path
        
    def load_keywords(self) -> List[str]:
        """加载查询关键词"""
        with open('scripts/keywords.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def load_existing_papers(self) -> None:
        """加载已有论文信息用于去重"""
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取所有arXiv ID
        arxiv_pattern = r'arxiv\.org/abs/(\d+\.\d+)'
        self.existing_papers = set(re.findall(arxiv_pattern, content, re.IGNORECASE))
    
    def query_new_papers(self) -> List[Dict]:
        """查询arXiv最新论文"""
        new_papers = []
        client = arxiv.Client()
        
        # 查询最近3天的论文
        for keyword in self.keywords:
            search_query = f'ti:"{keyword}" OR abs:"{keyword}"'
            search = arxiv.Search(
                query=search_query,
                max_results=100,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            try:
                for result in client.results(search):
                    # 检查是否在30天内提交
                    if result.published.date() < (datetime.now() - timedelta(days=30)).date():
                        continue
                    
                    # 提取arXiv ID
                    arxiv_id = result.entry_id.split('/')[-1].replace('v1', '').replace('v2', '')
                    
                    # 去重检查
                    if arxiv_id in self.existing_papers:
                        continue
                    
                    paper_info = {
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'arxiv_id': arxiv_id,
                        'pdf_url': result.pdf_url,
                        'year': result.published.year,
                        'summary': result.summary,
                        'primary_category': result.primary_category if result.primary_category else 'cs.IR'
                    }
                    #print(paper_info)
                    #exit(0)
                    # 检查是否属于生成式推荐
                    if self.is_generative_recommendation(paper_info):
                        new_papers.append(paper_info)
                        self.existing_papers.add(arxiv_id)  # 避免重复添加
                        
            except Exception as e:
                print(f"查询关键词 '{keyword}' 时出错: {e}")
                continue
                
        return new_papers
    
    def is_generative_recommendation(self, paper_info: Dict) -> bool:
        """判断论文是否属于生成式推荐领域"""
        title_lower = paper_info['title'].lower()
        summary_lower = paper_info['summary'].lower()
        
        return any(keyword in title_lower or keyword in summary_lower 
                  for keyword in self.keywords)
    
    def format_paper_entry(self, paper: Dict) -> str:
        """格式化论文条目"""
        year = paper['year']
        # 根据分类确定会议标记
        venue = self.determine_venue(paper)
        abs_url = paper['pdf_url'].replace('pdf','abs').replace('v1', '').replace('v2', '')
        entry = f"- `{venue}({year})`{paper['title']} **[[PDF]({abs_url})]**\n"
        
        return entry
    
    def determine_venue(self, paper: Dict) -> str:
        """根据论文信息确定会议/期刊"""
        # 这里可以根据实际需要进行更复杂的判断
        return "Arxiv"
    
    def update_readme(self, new_papers: List[Dict]) -> bool:
        """更新README.md文件"""
        if not new_papers:
            print("没有发现新论文")
            return False
            
        # 按时间排序
        #new_papers.sort(key=lambda x: x['year'], reverse=True)
        
        # 读取现有README内容
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到Generative Recommendation部分
        pattern = r'(### Generative Recommendation\n)(.*?)(?=\n###|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("未找到Generative Recommendation部分")
            return False
        
        # 构建新内容
        new_entries = []
        for paper in new_papers:
            new_entries.append(self.format_paper_entry(paper))
        
        updated_section = match.group(1) + ''.join(new_entries) + match.group(2)
        new_content = content.replace(match.group(0), updated_section)
        
        # 写回文件
        with open(self.paper_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"成功添加 {len(new_papers)} 篇新论文")
        return True

def main():
    updater = PaperUpdater(paper_path='README.md')
    updater.load_existing_papers()

    new_papers = updater.query_new_papers()
    os.makedirs(f'./new papers', exist_ok=True)
    # 保存新论文到JSON文件
    with open(f'./new papers/{datetime.now().strftime("%Y-%m-%d")}.json', 'w', encoding='utf-8') as f:
        json.dump(new_papers, f, ensure_ascii=False, indent=4)
        
    if updater.update_readme(new_papers):
        # 生成提交信息
        commit_message = f"Auto-update: Add {len(new_papers)} new papers - {datetime.now().strftime('%Y-%m-%d')}"
        print(commit_message)
    else:
        print("无需更新")

if __name__ == "__main__":
    main()


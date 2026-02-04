#!/usr/bin/env python3
"""
Disclosure Reports Crawler
==========================
Công cụ crawl và phân tích báo cáo lỗ hổng bảo mật đã được công khai từ Bugcrowd và HackerOne.

Author: dcduc168
Repository: https://github.com/dcduc168/DEG-WAF
"""

import requests
import json
import argparse
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Set
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup


# ============================================================================
# BASE CRAWLER CLASS
# ============================================================================

class BaseCrawler(ABC):
    """Base class cho tất cả crawlers với proxy support"""
    
    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Tạo session với proxy configuration nếu có"""
        session = requests.Session()
        
        if self.proxy:
            proxies = {
                'http': self.proxy,
                'https': self.proxy
            }
            session.proxies.update(proxies)
            print(f"🔧 Proxy enabled: {self.proxy}")
            
            # Disable SSL verification khi dùng proxy (Burp Suite)
            session.verify = False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        return session
    
    @abstractmethod
    def crawl(self, **kwargs) -> List[Dict]:
        """Crawl reports - phải được implement bởi subclass"""
        pass


# ============================================================================
# BUGCROWD CRAWLER
# ============================================================================

BUGCROWD_API_URL = "https://bugcrowd.com/crowdstream.json"

BUGCROWD_ATTACK_TYPES = {
    "sql": ["sql", "sqli", "injection"],
    "ssrf": ["ssrf", "server-side-request-forgery"],
    "command": ["command-injection", "cmd", "rce", "remote-code-execution"],
    "xss": ["xss", "cross-site-scripting", "stored-xss", "reflected-xss", "dom-xss"],
    "nosql": ["nosql", "mongodb", "no-sql"]
}


class BugcrowdCrawler(BaseCrawler):
    """Crawler cho Bugcrowd disclosure reports"""
    
    def __init__(self, proxy: Optional[str] = None):
        super().__init__(proxy)
        self.pagination_info = None
        self.metadata_file = "data/bugcrowd_metadata.json"
        self.cache_file = "data/bugcrowd_full_cache.json"
        self.metadata = self.load_metadata()
        self.cached_reports = self.load_cache()
    
    def load_metadata(self) -> Dict:
        """Load metadata từ lần crawl trước"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"last_crawl": None, "total_reports": 0, "last_report_id": None}
    
    def save_metadata(self):
        """Save metadata sau khi crawl"""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load_cache(self) -> List[Dict]:
        """Load full cache của reports"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def save_cache(self, reports: List[Dict]):
        """Save full cache của reports"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
    
    def fetch_page(self, page: int) -> tuple[List[Dict], Optional[Dict]]:
        """Fetch một trang từ Bugcrowd API"""
        params = {"page": page, "filter_by": "disclosures"}
        try:
            response = self.session.get(BUGCROWD_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if self.pagination_info is None:
                self.pagination_info = data.get("pagination_meta", {})
            
            return data.get("results", []), data.get("pagination_meta")
        except Exception as e:
            print(f"❌ Error fetching page {page}: {e}")
            return [], None
    
    def get_total_pages(self) -> int:
        """Auto-detect tổng số trang từ API"""
        if self.pagination_info is None:
            _, pagination = self.fetch_page(1)
            if pagination:
                self.pagination_info = pagination
        return self.pagination_info.get("total_pages", 31) if self.pagination_info else 31
    
    def check_for_updates(self) -> List[Dict]:
        """Crawl chỉ những reports mới (incremental update)"""
        print(f"🔄 [Bugcrowd] Checking for updates...")
        
        # Nếu chưa có cache, crawl toàn bộ
        if not self.cached_reports:
            print(f"   📦 No cache found, performing full crawl...")
            return self.crawl_full()
        
        # Tạo set của report IDs đã có
        existing_ids = {report.get("id") for report in self.cached_reports}
        print(f"   📊 Cache: {len(existing_ids)} reports")
        
        # Crawl từ page 1 cho đến khi gặp report đã có
        new_reports = []
        page = 1
        
        while True:
            print(f"   📄 Checking page {page}...", end=" ")
            reports, _ = self.fetch_page(page)
            
            if not reports:
                print("No data")
                break
            
            # Kiểm tra reports mới
            new_in_page = [r for r in reports if r.get("id") not in existing_ids]
            
            if not new_in_page:
                print(f"✓ No new reports (stopped)")
                break
            
            new_reports.extend(new_in_page)
            print(f"✓ Found {len(new_in_page)} new reports")
            
            # Nếu page này có report cũ, dừng lại
            if len(new_in_page) < len(reports):
                print(f"   🛑 Reached existing reports, stopping...")
                break
            
            page += 1
            
            # Safety limit
            if page > 10:
                print(f"   ⚠️  Reached page limit (10), stopping...")
                break
        
        if new_reports:
            print(f"\n✅ Found {len(new_reports)} new reports")
            # Merge với cache (new reports ở đầu)
            all_reports = new_reports + self.cached_reports
            self.save_cache(all_reports)
            self.cached_reports = all_reports
        else:
            print(f"\n✅ Cache is up-to-date (0 new reports)")
        
        return new_reports
    
    def crawl_full(self, max_pages: Optional[int] = None) -> List[Dict]:
        """Crawl toàn bộ reports (full crawl)"""
        if max_pages is None:
            max_pages = self.get_total_pages()
        
        print(f"🔄 [Bugcrowd] Full crawl: {max_pages} pages...")
        
        if self.pagination_info:
            total_count = self.pagination_info.get("totalCount", "?")
            limit = self.pagination_info.get("limit", "?")
            print(f"ℹ️  Total reports: {total_count} ({limit} per page)")
        
        all_reports = []
        for page in range(1, max_pages + 1):
            print(f"    📄 Page {page}/{max_pages}...", end=" ")
            reports, _ = self.fetch_page(page)
            
            if not reports:
                print("No data")
                break
            
            all_reports.extend(reports)
            print(f"✓ +{len(reports)} reports")
        
        print(f"\n✅ Crawled {len(all_reports)} reports")
        
        # Save to cache
        self.save_cache(all_reports)
        self.cached_reports = all_reports
        
        return all_reports
    
    def crawl(self, max_pages: Optional[int] = None, force_full: bool = False) -> List[Dict]:
        """Crawl reports từ Bugcrowd (smart incremental by default)"""
        if force_full or max_pages is not None:
            # Force full crawl nếu user chỉ định max_pages
            return self.crawl_full(max_pages)
        else:
            # Smart incremental update
            new_reports = self.check_for_updates()
            # Trả về toàn bộ cache (bao gồm cả reports cũ)
            return self.cached_reports
    
    def filter_by_attack_type(self, reports: List[Dict], attack_type: str,
                              min_severity: Optional[str] = None,
                              exclude_engagement: Optional[str] = None,
                              use_cache: bool = True) -> List[Dict]:
        """Lọc reports theo attack type và severity
        
        Args:
            reports: Danh sách reports (có thể bỏ qua nếu use_cache=True)
            attack_type: Loại tấn công
            min_severity: Severity tối thiểu
            exclude_engagement: Engagement code cần loại trừ
            use_cache: Nếu True, filter từ cache thay vì reports parameter
        """
        if attack_type not in BUGCROWD_ATTACK_TYPES:
            raise ValueError(f"Invalid attack type for Bugcrowd: {attack_type}")
        
        # Nếu use_cache, filter từ cached_reports
        if use_cache and self.cached_reports:
            print(f"   🔍 Filtering from cache ({len(self.cached_reports)} reports)...")
            reports = self.cached_reports
        
        keywords = BUGCROWD_ATTACK_TYPES[attack_type]
        filtered = []
        
        for report in reports:
            url = report.get("disclosure_report_url", "").lower()
            
            # Lọc theo keyword trong URL
            if not any(keyword in url for keyword in keywords):
                continue
            
            # Lọc theo severity (convert từ priority)
            if min_severity is not None:
                priority = report.get("priority")
                if priority is not None:
                    report_severity = PRIORITY_TO_SEVERITY.get(priority, "none")
                    severity_order = ["none", "low", "medium", "high", "critical"]
                    min_index = severity_order.index(min_severity.lower())
                    report_index = severity_order.index(report_severity)
                    if report_index < min_index:
                        continue
            
            # Loại trừ engagement code
            if exclude_engagement:
                engagement_code = report.get("engagement_code", "")
                if engagement_code == exclude_engagement:
                    continue
            
            filtered.append(report)
        
        return filtered
    
    def download_report_html(self, report: Dict, output_dir: str = "data/reports") -> Optional[str]:
        """Download HTML của một report từ Bugcrowd (chỉ lấy nội dung trong <article>)"""
        disclosure_path = report.get("disclosure_report_url")
        if not disclosure_path:
            return None
        
        # Tạo full URL từ path
        if not disclosure_path.startswith("http"):
            url = f"https://bugcrowd.com{disclosure_path}"
        else:
            url = disclosure_path
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Parse HTML và lấy nội dung trong <article>
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article')
            
            if not articles:
                print(f"      ⚠️  No <article> tags found")
                return None
            
            # Tạo HTML mới chỉ chứa các <article>
            article_html = "\n".join(str(article) for article in articles)
            
            # Tạo filename từ URL path
            filename = disclosure_path.split("/")[-1] + ".html"
            filepath = os.path.join(output_dir, filename)
            
            os.makedirs(output_dir, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(article_html)
            
            return filepath
        except Exception as e:
            print(f"      ❌ Error downloading {url}: {e}")
            return None


# ============================================================================
# HACKERONE CRAWLER
# ============================================================================

HACKERONE_GRAPHQL_URL = "https://hackerone.com/graphql"

HACKERONE_ATTACK_TYPES = {
    "xss": "Cross-Site Scripting (XSS)",
    "sqli": "SQL Injection",
    "ssrf": "Server-Side Request Forgery (SSRF)",
    "rce": "Code Injection",
    "csrf": "Cross-Site Request Forgery (CSRF)",
    "idor": "Insecure Direct Object References (IDOR)",
    "xxe": "Improper Restriction of XML External Entity Reference ('XXE')",
    "lfi": "Path Traversal",
    "open-redirect": "URL Redirection to Untrusted Site ('Open Redirect')"
}

# Mapping giữa priority levels và severity ratings
# Priority (Bugcrowd): 1 (cao nhất) -> 5 (thấp nhất)
# Severity (HackerOne): critical > high > medium > low > none
PRIORITY_TO_SEVERITY = {
    1: "critical",
    2: "high",
    3: "medium",
    4: "low",
    5: "none"
}

SEVERITY_TO_PRIORITY = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "none": 5
}

GRAPHQL_QUERY = """query HacktivitySearchQuery($queryString: String!, $from: Int, $size: Int, $sort: SortInput!) {
  me { id __typename }
  search(index: CompleteHacktivityReportIndex query_string: $queryString from: $from size: $size sort: $sort) {
    __typename total_count
    nodes {
      __typename
      ... on HacktivityDocument {
        id _id
        reporter { id username name __typename }
        cve_ids cwe severity_rating upvoted: upvoted_by_current_user public
        report {
          id databaseId: _id title substate url disclosed_at
          report_generated_content { id hacktivity_summary __typename }
          __typename
        }
        votes
        team { id handle name medium_profile_picture: profile_picture(size: medium) url currency __typename }
        total_awarded_amount latest_disclosable_action latest_disclosable_activity_at
        submitted_at disclosed has_collaboration
        collaborators { id username name __typename }
        __typename
      }
    }
  }
}"""


class HackerOneCrawler(BaseCrawler):
    """Crawler cho HackerOne disclosed reports"""
    
    def __init__(self, proxy: Optional[str] = None):
        super().__init__(proxy)
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.metadata_file = "data/hackerone_metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load metadata từ lần crawl trước"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"last_crawl": None, "total_reports": 0, "last_report_id": None}
    
    def save_metadata(self):
        """Save metadata sau khi crawl"""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def fetch_page(self, attack_type: str, page: int = 0, page_size: int = 100) -> tuple[List[Dict], int]:
        """Fetch một trang từ HackerOne GraphQL API"""
        cwe_name = HACKERONE_ATTACK_TYPES.get(attack_type, "")
        if not cwe_name:
            raise ValueError(f"Invalid attack type for HackerOne: {attack_type}")
        
        query_string = f'cwe:("{cwe_name}") AND disclosed:true'
        
        payload = {
            "operationName": "HacktivitySearchQuery",
            "variables": {
                "queryString": query_string,
                "size": page_size,
                "from": page * page_size,
                "sort": {"field": "latest_disclosable_activity_at", "direction": "DESC"},
                "product_area": "hacktivity",
                "product_feature": "overview"
            },
            "query": GRAPHQL_QUERY
        }
        
        try:
            response = self.session.post(HACKERONE_GRAPHQL_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            search_data = data.get("data", {}).get("search", {})
            total_count = search_data.get("total_count", 0)
            nodes = search_data.get("nodes", [])
            
            return nodes, total_count
        except Exception as e:
            print(f"❌ Error fetching page {page}: {e}")
            return [], 0
    
    def crawl(self, attack_type: str, max_reports: Optional[int] = None) -> List[Dict]:
        """Crawl reports từ HackerOne"""
        print(f"🔄 [HackerOne] Crawling {attack_type.upper()} reports...")
        
        first_page, total_count = self.fetch_page(attack_type, page=0)
        
        if not first_page:
            print("❌ No data received")
            return []
        
        print(f"ℹ️  Total reports: {total_count}")
        
        all_reports = first_page
        page_size = 100
        
        # Nếu có max_reports, limit số lượng cần fetch
        if max_reports:
            if len(all_reports) >= max_reports:
                all_reports = all_reports[:max_reports]
                print(f"\n✅ Crawled {len(all_reports)} reports (limited by --max)")
                return all_reports
            total_count = min(total_count, max_reports)
        
        total_pages = (total_count + page_size - 1) // page_size
        
        for page in range(1, total_pages):
            print(f"    📄 Page {page + 1}/{total_pages}...", end=" ")
            reports, _ = self.fetch_page(attack_type, page=page)
            
            if not reports:
                print("No data")
                break
            
            all_reports.extend(reports)
            print(f"✓ +{len(reports)} reports")
            
            if max_reports and len(all_reports) >= max_reports:
                all_reports = all_reports[:max_reports]
                break
        
        print(f"\n✅ Crawled {len(all_reports)} reports")
        return all_reports
    
    def filter_by_severity(self, reports: List[Dict], min_severity: Optional[str] = None) -> List[Dict]:
        """Lọc reports theo severity level"""
        if not min_severity:
            return reports
        
        severity_order = ["none", "low", "medium", "high", "critical"]
        min_index = severity_order.index(min_severity.lower())
        
        filtered = []
        for report in reports:
            severity = report.get("severity_rating", "").lower()
            if severity in severity_order:
                if severity_order.index(severity) >= min_index:
                    filtered.append(report)
        
        return filtered
    
    def download_report_html(self, report: Dict, output_dir: str = "data/reports") -> Optional[str]:
        """Download JSON của một report từ HackerOne"""
        report_data = report.get("report", {})
        report_id = report_data.get("databaseId")
        
        if not report_id:
            return None
        
        # HackerOne JSON API endpoint
        json_url = f"https://hackerone.com/reports/{report_id}.json"
        
        try:
            response = self.session.get(json_url, timeout=15)
            response.raise_for_status()
            
            # Tạo filename từ report ID
            filename = f"hackerone_{report_id}.json"
            filepath = os.path.join(output_dir, filename)
            
            os.makedirs(output_dir, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            
            return filepath
        except Exception as e:
            print(f"      ❌ Error downloading report {report_id}: {e}")
            return None


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_selection(selection: str, max_index: int) -> Set[int]:
    """Parse selection string như '1,2,5' hoặc '1-10' hoặc '1-2,5-7'"""
    indices = set()
    
    # Split bởi dấu phẩy
    parts = selection.split(",")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check nếu là range (1-10)
        if "-" in part:
            match = re.match(r"^(\d+)-(\d+)$", part)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                if start > end:
                    start, end = end, start
                for i in range(start, end + 1):
                    if 1 <= i <= max_index:
                        indices.add(i)
        else:
            # Single number
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= max_index:
                    indices.add(idx)
    
    return indices


def display_reports_table(reports: List[Dict], platform: str):
    """Hiển thị bảng danh sách reports"""
    print(f"\n{'='*100}")
    print(f"  REPORTS LIST ({len(reports)} items)")
    print(f"{'='*100}")
    print(f"{'No.':<6} {'Severity':<10} {'Title':<60} {'URL':<24}")
    print(f"{'-'*100}")
    
    for idx, report in enumerate(reports, 1):
        if platform == "bugcrowd":
            priority = report.get("priority", "?")
            severity = PRIORITY_TO_SEVERITY.get(priority, "unknown")
            title = report.get("title", "N/A")[:58]
            url = report.get("disclosure_report_url", "")[-22:]
        else:  # hackerone
            severity = report.get("severity_rating") or "unknown"
            report_data = report.get("report", {})
            title = report_data.get("title", "N/A")[:58]
            url = report_data.get("url", "")[-22:]
        
        print(f"{idx:<6} {severity.upper():<10} {title:<60} ...{url}")
    
    print(f"{'-'*100}")


def interactive_download(reports: List[Dict], crawler, platform: str):
    """Interactive mode để chọn và download reports"""
    display_reports_table(reports, platform)
    
    print(f"\n📥 SELECT REPORTS TO DOWNLOAD")
    print(f"   Examples: '1,2,5' or '1-10' or '1-2,5-7' or 'all'")
    print(f"   Press Enter to skip\n")
    
    selection = input("   Selection: ").strip()
    
    if not selection:
        print("   Skipped.")
        return
    
    # Handle 'all'
    if selection.lower() == "all":
        indices = set(range(1, len(reports) + 1))
    else:
        indices = parse_selection(selection, len(reports))
    
    if not indices:
        print("   ⚠️  No valid selection")
        return
    
    selected_reports = [reports[i-1] for i in sorted(indices)]
    
    print(f"\n📥 Downloading {len(selected_reports)} reports...")
    
    output_dir = f"data/reports/{platform}"
    success_count = 0
    
    for idx, report in enumerate(selected_reports, 1):
        if platform == "bugcrowd":
            title = report.get("title", "Unknown")[:50]
        else:
            title = report.get("report", {}).get("title", "Unknown")[:50]
        
        print(f"   [{idx}/{len(selected_reports)}] {title}...", end=" ")
        
        filepath = crawler.download_report_html(report, output_dir)
        
        if filepath:
            print(f"✓")
            success_count += 1
        else:
            print(f"✗")
        
        # Rate limiting
        if idx < len(selected_reports):
            time.sleep(0.5)
    
    print(f"\n✅ Downloaded {success_count}/{len(selected_reports)} reports to: {output_dir}")


def run_bugcrowd(args):
    """Execute Bugcrowd crawler với các options"""
    crawler = BugcrowdCrawler(proxy=args.proxy)
    
    # Kiểm tra force-full flag
    force_full = args.force_full if hasattr(args, 'force_full') else False
    
    # Crawl dữ liệu (smart incremental by default)
    reports = crawler.crawl(max_pages=args.max_pages, force_full=force_full)
    
    if not reports:
        print("⚠️  No data available")
        return
    
    print(f"\n📊 Total reports in cache: {len(reports)}")
    
    # Lọc theo attack type nếu có
    if args.attack:
        print(f"\n🔍 Filtering by attack type: {args.attack.upper()}")
        # Sử dụng cache để filter nhanh hơn
        reports = crawler.filter_by_attack_type(
            reports,
            args.attack,
            min_severity=args.filter,
            exclude_engagement=args.exclude,
            use_cache=True
        )
        print(f"   Remaining: {len(reports)} reports")
    
    # Lưu kết quả
    if args.attack:
        output_file = f"data/bugcrowd_{args.attack}_reports.json"
    else:
        output_file = "data/bugcrowd_reports.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    
    # Update metadata
    crawler.metadata["last_crawl"] = datetime.now().isoformat()
    crawler.metadata["total_reports"] = len(reports)
    if reports:
        crawler.metadata["last_report_id"] = reports[0].get("id")
    if crawler.pagination_info:
        crawler.metadata["pagination_info"] = crawler.pagination_info
    crawler.save_metadata()
    
    print(f"\n💾 Saved {len(reports)} reports to: {output_file}")
    
    # Interactive download nếu có --interactive
    if args.interactive and reports:
        interactive_download(reports, crawler, "bugcrowd")


def run_hackerone(args):
    """Execute HackerOne crawler với các options"""
    if not args.attack:
        print("❌ HackerOne requires --attack <type>")
        return
    
    crawler = HackerOneCrawler(proxy=args.proxy)
    
    # Crawl dữ liệu
    reports = crawler.crawl(args.attack, max_reports=args.max_reports)
    
    if not reports:
        print("⚠️  No data available")
        return
    
    # Lọc theo severity nếu có
    if args.filter:
        print(f"\n🔍 Filtering by severity >= {args.filter.upper()}")
        original_count = len(reports)
        reports = crawler.filter_by_severity(reports, args.filter)
        print(f"   Remaining: {len(reports)}/{original_count} reports")
    
    # Lưu kết quả
    output_file = f"data/hackerone_{args.attack}_reports.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    
    # Update metadata
    crawler.metadata["last_crawl"] = datetime.now().isoformat()
    crawler.metadata["total_reports"] = len(reports)
    crawler.metadata["attack_type"] = args.attack
    if reports:
        crawler.metadata["last_report_id"] = reports[0].get("id")
    crawler.save_metadata()
    
    # Thống kê
    print(f"\n📊 Statistics:")
    print(f"   Total reports: {len(reports)}")
    
    severity_count = {}
    for report in reports:
        severity = report.get("severity_rating", "unknown")
        severity_count[severity] = severity_count.get(severity, 0) + 1
    
    print(f"   Severity distribution:")
    for severity in ["critical", "high", "medium", "low", "none"]:
        count = severity_count.get(severity, 0)
        if count > 0:
            print(f"      • {severity.upper()}: {count}")
    
    print(f"\n💾 Saved {len(reports)} reports to: {output_file}")
    
    # Interactive download nếu có --interactive
    if args.interactive and reports:
        interactive_download(reports, crawler, "hackerone")


def main():
    parser = argparse.ArgumentParser(
        prog="disclosure-crawler",
        description="Disclosure Reports Crawler - Crawl vulnerability reports from Bugcrowd & HackerOne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
─────────────────────────────────────────────────────────────────────────────

Bugcrowd Platform:
  
  • Check for updates (smart incremental)
    $ python disclosure-crawler.py --platform bugcrowd
  
  • Filter XSS from cache (no re-crawl needed)
    $ python disclosure-crawler.py --platform bugcrowd --attack xss --filter high
  
  • Force full crawl (10 pages)
    $ python disclosure-crawler.py --platform bugcrowd --max-pages 10
  
  • Force complete re-crawl
    $ python disclosure-crawler.py --platform bugcrowd --force-full
  
  • Exclude specific engagement
    $ python disclosure-crawler.py --platform bugcrowd --attack sql --exclude nasa-vdp

HackerOne Platform:
  
  • Crawl XSS reports
    $ python disclosure-crawler.py --platform hackerone --attack xss
  
  • Crawl SQL Injection (limit to 50 reports)
    $ python disclosure-crawler.py --platform hackerone --attack sqli --max 50
  
  • Crawl SSRF with high+ severity filter
    $ python disclosure-crawler.py --platform hackerone --attack ssrf --filter high
  
  • Crawl critical RCE reports only
    $ python disclosure-crawler.py --platform hackerone --attack rce --filter critical

Interactive Download:
  
  • List reports and select which to download
    $ python disclosure-crawler.py --platform bugcrowd --attack xss -i
    $ python disclosure-crawler.py --platform hackerone --attack sqli --interactive
  
  Selection examples:
    1,2,5        → Download reports #1, #2, #5
    1-10         → Download reports #1 through #10
    1-5,8,10-12  → Download reports #1-5, #8, #10-12
    all          → Download all reports

Debugging with Proxy:
  
  • Use Burp Suite for traffic inspection
    $ python disclosure-crawler.py --platform hackerone --attack xss --proxy http://127.0.0.1:8080

NOTES:
─────────────────────────────────────────────────────────────────────────────

Severity Filter (--filter):
  Works on both platforms with automatic priority↔severity mapping:
  
  Bugcrowd Priority → Severity:
    Priority 1 → critical
    Priority 2 → high
    Priority 3 → medium
    Priority 4 → low
    Priority 5 → none
  
  HackerOne: Uses severity directly from API
  
Attack Types:
  
  Bugcrowd:  sql, ssrf, command, xss, nosql
  HackerOne: xss, sqli, ssrf, rce, csrf, idor, xxe, lfi, open-redirect

Output:
  
  JSON reports: data/<platform>_<attack>_reports.json
  Downloaded reports: data/reports/bugcrowd/*.html (Bugcrowd)
                      data/reports/hackerone/*.json (HackerOne)
  Metadata: data/<platform>_metadata.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=["bugcrowd", "hackerone"],
        metavar="PLATFORM",
        help="target platform (bugcrowd | hackerone)"
    )
    
    # Common arguments
    parser.add_argument(
        "--attack",
        type=str,
        metavar="TYPE",
        help="attack type to filter (required for HackerOne)"
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        choices=["low", "medium", "high", "critical"],
        metavar="SEVERITY",
        help="minimum severity level (low | medium | high | critical)"
    )
    
    parser.add_argument(
        "--proxy",
        type=str,
        metavar="URL",
        help="HTTP proxy for debugging (e.g., http://127.0.0.1:8080)"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="interactive mode to select and download reports (HTML for Bugcrowd, JSON for HackerOne)"
    )
    
    # Bugcrowd-specific arguments
    bugcrowd_group = parser.add_argument_group("Bugcrowd Options")
    bugcrowd_group.add_argument(
        "--max-pages",
        type=int,
        metavar="N",
        help="maximum number of pages to crawl (forces full crawl)"
    )
    
    bugcrowd_group.add_argument(
        "--force-full",
        action="store_true",
        help="force full crawl instead of incremental update"
    )
    
    bugcrowd_group.add_argument(
        "--exclude",
        type=str,
        metavar="CODE",
        help="exclude specific engagement code (e.g., nasa-vdp)"
    )
    
    # HackerOne-specific arguments
    hackerone_group = parser.add_argument_group("HackerOne Options")
    hackerone_group.add_argument(
        "--max",
        dest="max_reports",
        type=int,
        metavar="N",
        help="maximum number of reports to fetch"
    )
    
    args = parser.parse_args()
    
    # Validate attack types
    if args.attack:
        if args.platform == "bugcrowd":
            if args.attack not in BUGCROWD_ATTACK_TYPES:
                valid_types = ", ".join(sorted(BUGCROWD_ATTACK_TYPES.keys()))
                parser.error(f"Invalid attack type for Bugcrowd. Valid types: {valid_types}")
        elif args.platform == "hackerone":
            if args.attack not in HACKERONE_ATTACK_TYPES:
                valid_types = ", ".join(sorted(HACKERONE_ATTACK_TYPES.keys()))
                parser.error(f"Invalid attack type for HackerOne. Valid types: {valid_types}")
    
    # HackerOne requires attack type
    if args.platform == "hackerone" and not args.attack:
        parser.error("HackerOne platform requires --attack <type>")
    
    # Execute crawler
    print(f"{'='*80}")
    print(f"  Disclosure Reports Crawler")
    print(f"  Platform: {args.platform.upper()}")
    if args.attack:
        print(f"  Attack Type: {args.attack.upper()}")
    if args.filter:
        print(f"  Min Severity: {args.filter.upper()}")
    print(f"{'='*80}\n")
    
    if args.platform == "bugcrowd":
        run_bugcrowd(args)
    elif args.platform == "hackerone":
        run_hackerone(args)


if __name__ == "__main__":
    main()

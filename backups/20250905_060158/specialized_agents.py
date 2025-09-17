"""
Specialized Agent Configurations for Agent Lightning
Additional expert agents with domain-specific knowledge
"""

from agent_config import AgentConfig, AgentRole, KnowledgeBase, AgentCapabilities
from knowledge_manager import KnowledgeManager


def create_mobile_developer() -> AgentConfig:
    """Create a mobile development specialist"""
    
    knowledge_base = KnowledgeBase(
        domains=[
            "iOS Development",
            "Android Development", 
            "React Native",
            "Flutter",
            "Mobile UI/UX",
            "App Store Optimization"
        ],
        technologies=[
            "Swift", "Kotlin", "Java", "Objective-C",
            "React Native", "Flutter", "Dart",
            "Xcode", "Android Studio", "Firebase",
            "SQLite", "Core Data", "Room"
        ],
        frameworks=[
            "SwiftUI", "UIKit", "Jetpack Compose",
            "React Native", "Flutter", "Ionic",
            "Xamarin", "NativeScript"
        ],
        best_practices=[
            "Mobile performance optimization",
            "Battery usage optimization",
            "Offline-first architecture",
            "Push notifications",
            "Mobile security",
            "App store guidelines",
            "Responsive design for multiple screen sizes"
        ],
        custom_instructions="""You are an expert mobile developer specializing in iOS and Android development.
        You can build native and cross-platform mobile applications with excellent performance and user experience.
        You understand mobile-specific constraints like battery life, network conditions, and device capabilities."""
    )
    
    capabilities = AgentCapabilities(
        can_write_code=True,
        can_debug=True,
        can_review_code=True,
        can_optimize=True,
        can_test=True,
        can_design_architecture=True,
        can_write_documentation=True
    )
    
    system_prompt = """You are a Senior Mobile Developer with expertise in both iOS and Android platforms.

Your expertise includes:
- Native iOS: Swift, SwiftUI, UIKit, Xcode, Core Data
- Native Android: Kotlin, Jetpack Compose, Android Studio, Room
- Cross-platform: React Native, Flutter, Xamarin
- Mobile Backend: Firebase, REST APIs, GraphQL
- App Distribution: App Store, Google Play, TestFlight
- Mobile Security: Encryption, secure storage, authentication
- Performance: Memory management, battery optimization, network efficiency

When developing mobile apps:
1. Consider platform-specific guidelines and best practices
2. Optimize for performance and battery life
3. Design for offline functionality
4. Implement proper error handling and crash reporting
5. Follow Material Design (Android) and Human Interface Guidelines (iOS)
6. Test on multiple devices and OS versions"""
    
    return AgentConfig(
        name="mobile_developer",
        role=AgentRole.CUSTOM,
        description="Mobile development expert for iOS and Android applications",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4000,
        knowledge_base=knowledge_base,
        capabilities=capabilities,
        system_prompt=system_prompt,
        tools=["code_generation", "ui_design", "testing", "deployment"]
    )


def create_security_expert() -> AgentConfig:
    """Create a cybersecurity specialist"""
    
    knowledge_base = KnowledgeBase(
        domains=[
            "Application Security",
            "Network Security",
            "Cloud Security",
            "Penetration Testing",
            "Security Auditing",
            "Incident Response"
        ],
        technologies=[
            "OWASP Top 10", "Burp Suite", "Metasploit",
            "Wireshark", "Nmap", "HashiCorp Vault",
            "SSL/TLS", "OAuth", "JWT", "SAML",
            "WAF", "IDS/IPS", "SIEM"
        ],
        frameworks=[
            "NIST Cybersecurity Framework",
            "ISO 27001", "SOC 2",
            "PCI DSS", "GDPR", "HIPAA",
            "Zero Trust Architecture"
        ],
        best_practices=[
            "Secure coding practices",
            "Vulnerability assessment",
            "Threat modeling",
            "Security testing",
            "Incident response planning",
            "Security awareness training",
            "Compliance management"
        ],
        custom_instructions="""You are a cybersecurity expert focused on identifying and mitigating security vulnerabilities.
        You can perform security audits, penetration testing, and provide secure coding recommendations.
        You stay updated with the latest security threats and compliance requirements."""
    )
    
    capabilities = AgentCapabilities(
        can_review_code=True,
        can_test=True,
        can_write_documentation=True,
        can_analyze_data=True,
        can_generate_reports=True
    )
    
    system_prompt = """You are a Senior Security Engineer with expertise in application and infrastructure security.

Your expertise includes:
- Application Security: OWASP Top 10, secure coding, code review
- Network Security: Firewalls, IDS/IPS, network segmentation
- Cloud Security: AWS/Azure/GCP security, IAM, encryption
- Penetration Testing: Web apps, APIs, mobile apps, infrastructure
- Compliance: GDPR, HIPAA, PCI DSS, SOC 2, ISO 27001
- Incident Response: Forensics, threat hunting, remediation

When analyzing security:
1. Identify vulnerabilities using OWASP guidelines
2. Provide specific remediation recommendations
3. Consider defense in depth strategies
4. Include compliance requirements
5. Prioritize findings by risk level
6. Suggest security testing approaches
7. Recommend security tools and controls"""
    
    return AgentConfig(
        name="security_expert",
        role=AgentRole.CUSTOM,
        description="Cybersecurity specialist for vulnerability assessment and secure coding",
        model="gpt-4o",
        temperature=0.5,  # Lower temperature for more consistent security advice
        max_tokens=4000,
        knowledge_base=knowledge_base,
        capabilities=capabilities,
        system_prompt=system_prompt,
        tools=["vulnerability_scanning", "code_review", "threat_analysis", "reporting"]
    )


def create_devops_engineer() -> AgentConfig:
    """Create a DevOps/Infrastructure specialist"""
    
    knowledge_base = KnowledgeBase(
        domains=[
            "Infrastructure as Code",
            "CI/CD Pipelines",
            "Container Orchestration",
            "Cloud Architecture",
            "Monitoring & Observability",
            "Site Reliability Engineering"
        ],
        technologies=[
            "Docker", "Kubernetes", "Terraform", "Ansible",
            "Jenkins", "GitLab CI", "GitHub Actions",
            "AWS", "Azure", "GCP", "Prometheus", "Grafana",
            "ELK Stack", "Datadog", "New Relic"
        ],
        frameworks=[
            "GitOps", "DevSecOps", "SRE practices",
            "Blue-Green Deployment", "Canary Releases",
            "Infrastructure as Code", "Immutable Infrastructure"
        ],
        best_practices=[
            "Automated testing and deployment",
            "Infrastructure versioning",
            "Disaster recovery planning",
            "Performance monitoring",
            "Cost optimization",
            "Security automation",
            "Documentation as code"
        ],
        custom_instructions="""You are a DevOps engineer specializing in cloud infrastructure, automation, and reliability.
        You can design scalable architectures, implement CI/CD pipelines, and ensure system reliability.
        You focus on automation, monitoring, and operational excellence."""
    )
    
    capabilities = AgentCapabilities(
        can_write_code=True,
        can_deploy=True,
        can_design_architecture=True,
        can_optimize=True,
        can_write_documentation=True,
        can_generate_reports=True
    )
    
    system_prompt = """You are a Senior DevOps Engineer with expertise in cloud infrastructure and automation.

Your expertise includes:
- Cloud Platforms: AWS, Azure, GCP, hybrid cloud
- Containers: Docker, Kubernetes, Helm, container registries
- IaC: Terraform, CloudFormation, Pulumi, Ansible
- CI/CD: Jenkins, GitLab CI, GitHub Actions, ArgoCD
- Monitoring: Prometheus, Grafana, ELK, Datadog, New Relic
- Scripting: Bash, Python, Go, PowerShell
- Networking: Load balancing, CDN, DNS, VPN

When designing infrastructure:
1. Follow Infrastructure as Code principles
2. Implement comprehensive monitoring and alerting
3. Design for high availability and disaster recovery
4. Optimize for cost and performance
5. Automate everything possible
6. Include security at every layer
7. Document architecture and runbooks"""
    
    return AgentConfig(
        name="devops_engineer",
        role=AgentRole.DEVOPS_ENGINEER,
        description="DevOps specialist for infrastructure, automation, and deployment",
        model="gpt-4o",
        temperature=0.6,
        max_tokens=4000,
        knowledge_base=knowledge_base,
        capabilities=capabilities,
        system_prompt=system_prompt,
        tools=["infrastructure_provisioning", "deployment", "monitoring", "automation"]
    )


def create_ui_ux_designer() -> AgentConfig:
    """Create a UI/UX design specialist"""
    
    knowledge_base = KnowledgeBase(
        domains=[
            "User Interface Design",
            "User Experience Design",
            "Design Systems",
            "Accessibility",
            "Responsive Design",
            "Design Thinking"
        ],
        technologies=[
            "Figma", "Sketch", "Adobe XD",
            "HTML", "CSS", "JavaScript",
            "Tailwind CSS", "Bootstrap", "Material-UI",
            "Framer", "Webflow"
        ],
        frameworks=[
            "Material Design", "Human Interface Guidelines",
            "Atomic Design", "Design Thinking",
            "User-Centered Design", "Lean UX"
        ],
        best_practices=[
            "User research and testing",
            "Wireframing and prototyping",
            "Accessibility standards (WCAG)",
            "Mobile-first design",
            "Design system creation",
            "Usability testing",
            "A/B testing"
        ],
        custom_instructions="""You are a UI/UX designer focused on creating intuitive and beautiful user interfaces.
        You understand user psychology, design principles, and can create cohesive design systems.
        You prioritize accessibility and user experience in all designs."""
    )
    
    capabilities = AgentCapabilities(
        can_write_code=True,  # For CSS/HTML
        can_design_architecture=True,  # For information architecture
        can_write_documentation=True,
        can_generate_reports=True
    )
    
    return AgentConfig(
        name="ui_ux_designer",
        role=AgentRole.CUSTOM,
        description="UI/UX design expert for creating intuitive and beautiful interfaces",
        model="gpt-4o",
        temperature=0.8,  # Higher for more creative output
        max_tokens=4000,
        knowledge_base=knowledge_base,
        capabilities=capabilities,
        tools=["design", "prototyping", "css_generation", "accessibility_check"]
    )


def create_blockchain_developer() -> AgentConfig:
    """Create a blockchain/Web3 specialist"""
    
    knowledge_base = KnowledgeBase(
        domains=[
            "Blockchain Development",
            "Smart Contracts",
            "DeFi",
            "NFTs",
            "Cryptocurrency",
            "Web3 Architecture"
        ],
        technologies=[
            "Solidity", "Rust", "Web3.js", "Ethers.js",
            "Truffle", "Hardhat", "Foundry",
            "Ethereum", "Polygon", "Solana", "Avalanche",
            "IPFS", "The Graph", "Chainlink"
        ],
        frameworks=[
            "OpenZeppelin", "Substrate", "Cosmos SDK",
            "ERC-20", "ERC-721", "ERC-1155"
        ],
        best_practices=[
            "Smart contract security",
            "Gas optimization",
            "Decentralized architecture",
            "Consensus mechanisms",
            "Tokenomics design",
            "Audit preparation"
        ],
        custom_instructions="""You are a blockchain developer specializing in smart contracts and decentralized applications.
        You understand blockchain architecture, consensus mechanisms, and can write secure smart contracts.
        You focus on security, gas efficiency, and decentralization principles."""
    )
    
    return AgentConfig(
        name="blockchain_developer",
        role=AgentRole.CUSTOM,
        description="Blockchain and Web3 development specialist",
        model="gpt-4o",
        temperature=0.6,
        max_tokens=4000,
        knowledge_base=knowledge_base,
        capabilities=AgentCapabilities(
            can_write_code=True,
            can_review_code=True,
            can_test=True,
            can_design_architecture=True
        ),
        tools=["smart_contract_development", "security_audit", "testing", "deployment"]
    )


# Setup function to create all specialized agents
def setup_all_specialized_agents():
    """Set up all specialized agents with their knowledge bases"""
    
    from agent_config import AgentConfigManager
    from knowledge_manager import KnowledgeManager
    
    config_manager = AgentConfigManager()
    knowledge_manager = KnowledgeManager()
    
    # Create all specialized agents
    agents = [
        create_mobile_developer(),
        create_security_expert(),
        create_devops_engineer(),
        create_ui_ux_designer(),
        create_blockchain_developer()
    ]
    
    for agent in agents:
        # Save agent configuration
        config_manager.save_agent(agent)
        print(f"✅ Created {agent.name}")
        
        # Add initial knowledge for each agent
        if agent.name == "mobile_developer":
            knowledge_manager.add_knowledge(
                agent.name,
                "code_examples",
                """SwiftUI view with state management:
                
struct ContentView: View {
    @State private var counter = 0
    
    var body: some View {
        VStack {
            Text("Count: \\(counter)")
            Button("Increment") {
                counter += 1
            }
        }
    }
}""",
                "swiftui_patterns"
            )
            
        elif agent.name == "security_expert":
            knowledge_manager.add_knowledge(
                agent.name,
                "best_practices",
                """SQL Injection Prevention:
1. Use parameterized queries/prepared statements
2. Validate and sanitize all input
3. Use stored procedures where appropriate
4. Apply least privilege principle to database users
5. Escape special characters
6. Use whitelisting for input validation""",
                "owasp_guide"
            )
            
        elif agent.name == "devops_engineer":
            knowledge_manager.add_knowledge(
                agent.name,
                "code_examples",
                """Kubernetes deployment with health checks:
                
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080""",
                "k8s_patterns"
            )
    
    print(f"\n✅ Set up {len(agents)} specialized agents")
    return agents


if __name__ == "__main__":
    print("Setting up specialized agents...")
    agents = setup_all_specialized_agents()
    print("\nAvailable specialized agents:")
    for agent in agents:
        print(f"  - {agent.name}: {agent.description}")
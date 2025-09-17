#!/usr/bin/env python3
"""
Tech Stack Knowledge Base
Comprehensive knowledge about various technologies and frameworks
"""

from knowledge_manager import KnowledgeManager
from agent_config import AgentConfigManager


class TechStackKnowledge:
    """Add comprehensive tech stack knowledge to agents"""
    
    def __init__(self):
        self.km = KnowledgeManager()
        self.config_manager = AgentConfigManager()
    
    def add_java_knowledge(self, agent_name: str):
        """Add Java ecosystem knowledge"""
        
        java_knowledge = [
            {
                "category": "code_examples",
                "content": """Java Spring Boot REST Controller:

@RestController
@RequestMapping("/api/users")
public class UserController {
    @Autowired
    private UserService userService;
    
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        return userService.findById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody @Valid UserDto userDto) {
        User user = userService.create(userDto);
        return ResponseEntity.created(URI.create("/api/users/" + user.getId())).body(user);
    }
}""",
                "source": "java_spring_patterns"
            },
            {
                "category": "best_practices",
                "content": """Java Best Practices:
1. Use Optional instead of null returns
2. Prefer composition over inheritance
3. Use try-with-resources for AutoCloseable resources
4. Follow SOLID principles
5. Use meaningful variable and method names
6. Implement proper exception handling
7. Use Java 8+ features (Streams, Lambdas, Optional)
8. Write unit tests with JUnit and Mockito
9. Use dependency injection (Spring DI)
10. Follow Java naming conventions""",
                "source": "java_best_practices"
            },
            {
                "category": "code_examples",
                "content": """Java Microservice with Spring Cloud:

@SpringBootApplication
@EnableEurekaClient
@EnableCircuitBreaker
public class PaymentServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(PaymentServiceApplication.class, args);
    }
}

@Service
public class PaymentService {
    @Autowired
    private RestTemplate restTemplate;
    
    @HystrixCommand(fallbackMethod = "processPaymentFallback")
    public PaymentResult processPayment(PaymentRequest request) {
        // Main payment logic
        return restTemplate.postForObject(
            "http://payment-gateway/process",
            request,
            PaymentResult.class
        );
    }
    
    public PaymentResult processPaymentFallback(PaymentRequest request) {
        return new PaymentResult("PENDING", "Service temporarily unavailable");
    }
}""",
                "source": "java_microservices"
            },
            {
                "category": "architecture_patterns",
                "content": """Java Design Patterns:

1. Singleton Pattern:
public class DatabaseConnection {
    private static volatile DatabaseConnection instance;
    
    private DatabaseConnection() {}
    
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
}

2. Builder Pattern:
User user = User.builder()
    .name("John Doe")
    .email("john@example.com")
    .age(30)
    .build();

3. Factory Pattern
4. Observer Pattern
5. Strategy Pattern""",
                "source": "java_patterns"
            }
        ]
        
        for item in java_knowledge:
            self.km.add_knowledge(agent_name, item["category"], item["content"], item["source"])
        
        print(f"✅ Added {len(java_knowledge)} Java knowledge items to {agent_name}")
    
    def add_go_knowledge(self, agent_name: str):
        """Add Go language knowledge"""
        
        go_knowledge = [
            {
                "category": "code_examples",
                "content": """Go HTTP Server with Middleware:

package main

import (
    "encoding/json"
    "log"
    "net/http"
    "time"
    
    "github.com/gorilla/mux"
)

type User struct {
    ID        string    `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL)
        next.ServeHTTP(w, r)
    })
}

func getUserHandler(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    user := User{
        ID:        vars["id"],
        Name:      "John Doe",
        Email:     "john@example.com",
        CreatedAt: time.Now(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func main() {
    r := mux.NewRouter()
    r.Use(loggingMiddleware)
    r.HandleFunc("/users/{id}", getUserHandler).Methods("GET")
    
    srv := &http.Server{
        Handler:      r,
        Addr:         ":8080",
        WriteTimeout: 15 * time.Second,
        ReadTimeout:  15 * time.Second,
    }
    
    log.Fatal(srv.ListenAndServe())
}""",
                "source": "go_web_patterns"
            },
            {
                "category": "best_practices",
                "content": """Go Best Practices:
1. Handle errors explicitly - don't ignore them
2. Use goroutines and channels for concurrency
3. Keep interfaces small and focused
4. Use defer for cleanup operations
5. Avoid global variables
6. Use context for cancellation and timeouts
7. Write table-driven tests
8. Use go fmt and go vet
9. Document exported functions
10. Return early to reduce nesting
11. Use meaningful package names
12. Prefer composition over inheritance""",
                "source": "go_best_practices"
            },
            {
                "category": "code_examples",
                "content": """Go Concurrency Patterns:

// Worker Pool Pattern
func workerPool(jobs <-chan int, results chan<- int) {
    for j := range jobs {
        results <- j * 2
    }
}

func main() {
    numJobs := 100
    jobs := make(chan int, numJobs)
    results := make(chan int, numJobs)
    
    // Start workers
    for w := 1; w <= 3; w++ {
        go workerPool(jobs, results)
    }
    
    // Send jobs
    for j := 1; j <= numJobs; j++ {
        jobs <- j
    }
    close(jobs)
    
    // Collect results
    for a := 1; a <= numJobs; a++ {
        <-results
    }
}

// Context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

select {
case <-ctx.Done():
    return ctx.Err()
case result := <-ch:
    return result
}""",
                "source": "go_concurrency"
            }
        ]
        
        for item in go_knowledge:
            self.km.add_knowledge(agent_name, item["category"], item["content"], item["source"])
        
        print(f"✅ Added {len(go_knowledge)} Go knowledge items to {agent_name}")
    
    def add_hyperledger_fabric_knowledge(self, agent_name: str):
        """Add Hyperledger Fabric blockchain knowledge"""
        
        fabric_knowledge = [
            {
                "category": "code_examples",
                "content": """Hyperledger Fabric Chaincode (Smart Contract) in Go:

package main

import (
    "encoding/json"
    "fmt"
    
    "github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct {
    contractapi.Contract
}

type Asset struct {
    ID             string `json:"ID"`
    Owner          string `json:"owner"`
    Value          int    `json:"value"`
    AppraisedValue int    `json:"appraisedValue"`
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
    assets := []Asset{
        {ID: "asset1", Owner: "Alice", Value: 100, AppraisedValue: 150},
        {ID: "asset2", Owner: "Bob", Value: 200, AppraisedValue: 250},
    }
    
    for _, asset := range assets {
        assetJSON, err := json.Marshal(asset)
        if err != nil {
            return err
        }
        
        err = ctx.GetStub().PutState(asset.ID, assetJSON)
        if err != nil {
            return fmt.Errorf("failed to put to world state: %v", err)
        }
    }
    
    return nil
}

func (s *SmartContract) CreateAsset(ctx contractapi.TransactionContextInterface, 
    id string, owner string, value int, appraisedValue int) error {
    
    exists, err := s.AssetExists(ctx, id)
    if err != nil {
        return err
    }
    if exists {
        return fmt.Errorf("asset %s already exists", id)
    }
    
    asset := Asset{
        ID:             id,
        Owner:          owner,
        Value:          value,
        AppraisedValue: appraisedValue,
    }
    
    assetJSON, err := json.Marshal(asset)
    if err != nil {
        return err
    }
    
    return ctx.GetStub().PutState(id, assetJSON)
}

func (s *SmartContract) TransferAsset(ctx contractapi.TransactionContextInterface, 
    id string, newOwner string) error {
    
    asset, err := s.ReadAsset(ctx, id)
    if err != nil {
        return err
    }
    
    asset.Owner = newOwner
    assetJSON, err := json.Marshal(asset)
    if err != nil {
        return err
    }
    
    return ctx.GetStub().PutState(id, assetJSON)
}""",
                "source": "hyperledger_chaincode"
            },
            {
                "category": "architecture_patterns",
                "content": """Hyperledger Fabric Architecture:

1. Network Components:
   - Peers (Endorsing, Committing, Anchor)
   - Orderer nodes (Raft/Kafka consensus)
   - Certificate Authority (CA)
   - Channels for privacy
   - Organizations and MSP

2. Transaction Flow:
   1. Client submits proposal to endorsing peers
   2. Peers execute chaincode and return endorsement
   3. Client collects endorsements per policy
   4. Client submits to orderer
   5. Orderer creates blocks
   6. Blocks distributed to all peers
   7. Peers validate and commit

3. Key Concepts:
   - Chaincode (Smart Contracts)
   - Ledger (World State + Blockchain)
   - Private Data Collections
   - Endorsement Policies
   - Channel Configuration
   - Identity Management (MSP)""",
                "source": "fabric_architecture"
            },
            {
                "category": "code_examples",
                "content": """Fabric Network Configuration (configtx.yaml):

Organizations:
  - &OrdererOrg
    Name: OrdererOrg
    ID: OrdererMSP
    MSPDir: crypto-config/ordererOrganizations/example.com/msp
    
  - &Org1
    Name: Org1MSP
    ID: Org1MSP
    MSPDir: crypto-config/peerOrganizations/org1.example.com/msp
    AnchorPeers:
      - Host: peer0.org1.example.com
        Port: 7051

Capabilities:
  Channel: &ChannelCapabilities
    V2_0: true
  Orderer: &OrdererCapabilities
    V2_0: true
  Application: &ApplicationCapabilities
    V2_0: true

Application: &ApplicationDefaults
  Organizations:
  Policies:
    Readers:
      Type: ImplicitMeta
      Rule: "ANY Readers"
    Writers:
      Type: ImplicitMeta
      Rule: "ANY Writers"
    Admins:
      Type: ImplicitMeta
      Rule: "MAJORITY Admins"

Orderer: &OrdererDefaults
  OrdererType: etcdraft
  Addresses:
    - orderer.example.com:7050
  BatchTimeout: 2s
  BatchSize:
    MaxMessageCount: 10
    AbsoluteMaxBytes: 99 MB
    PreferredMaxBytes: 512 KB""",
                "source": "fabric_config"
            }
        ]
        
        for item in fabric_knowledge:
            self.km.add_knowledge(agent_name, item["category"], item["content"], item["source"])
        
        print(f"✅ Added {len(fabric_knowledge)} Hyperledger Fabric knowledge items to {agent_name}")
    
    def add_python_knowledge(self, agent_name: str):
        """Add Python ecosystem knowledge"""
        
        python_knowledge = [
            {
                "category": "code_examples",
                "content": """Python FastAPI Advanced Features:

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
from datetime import datetime, timedelta

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency Injection
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = await decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return user

# Background Tasks
def send_email(email: str, message: str):
    # Email sending logic
    pass

@app.post("/send-notification/")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    background_tasks.add_task(send_email, email, f"Hello {current_user.name}")
    return {"message": "Notification sent"}

# WebSocket Support
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(client_id)

# Async Database Operations
@app.get("/items/", response_model=List[Item])
async def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = await db.execute(
        select(Item).offset(skip).limit(limit)
    )
    return items.scalars().all()""",
                "source": "python_fastapi_advanced"
            },
            {
                "category": "best_practices",
                "content": """Python Best Practices:
1. Follow PEP 8 style guide
2. Use type hints (Python 3.5+)
3. Write docstrings for all functions/classes
4. Use virtual environments (venv, conda)
5. Handle exceptions properly
6. Use list comprehensions wisely
7. Avoid mutable default arguments
8. Use context managers (with statement)
9. Write unit tests with pytest
10. Use async/await for I/O operations
11. Profile code for performance
12. Use logging instead of print
13. Keep requirements.txt updated
14. Use dataclasses for data structures
15. Leverage itertools and functools""",
                "source": "python_best_practices"
            },
            {
                "category": "code_examples",
                "content": """Python Data Science Stack:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading and Preprocessing
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.fillna(df.mean())

# Feature Engineering
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Train-Test Split
X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluation
y_pred = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.show()""",
                "source": "python_data_science"
            }
        ]
        
        for item in python_knowledge:
            self.km.add_knowledge(agent_name, item["category"], item["content"], item["source"])
        
        print(f"✅ Added {len(python_knowledge)} Python knowledge items to {agent_name}")
    
    def add_react_knowledge(self, agent_name: str):
        """Add React ecosystem knowledge"""
        
        react_knowledge = [
            {
                "category": "code_examples",
                "content": """Modern React with Hooks and TypeScript:

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import axios from 'axios';

interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

interface UserListProps {
  filterRole?: string;
}

export const UserList: React.FC<UserListProps> = ({ filterRole }) => {
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  
  // React Query for data fetching
  const { data: users, isLoading, error } = useQuery({
    queryKey: ['users', filterRole],
    queryFn: async () => {
      const { data } = await axios.get<User[]>('/api/users', {
        params: { role: filterRole }
      });
      return data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
  
  // Mutation for updating user
  const updateUserMutation = useMutation({
    mutationFn: async (user: User) => {
      const { data } = await axios.put(`/api/users/${user.id}`, user);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
  
  // Memoized filtered users
  const filteredUsers = useMemo(() => {
    if (!users) return [];
    return users.filter(user => 
      user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [users, searchTerm]);
  
  // Callback to prevent recreation
  const handleUserSelect = useCallback((user: User) => {
    setSelectedUser(user);
  }, []);
  
  // Cleanup effect
  useEffect(() => {
    return () => {
      // Cleanup logic
    };
  }, []);
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return (
    <div className="user-list">
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search users..."
        className="search-input"
      />
      
      <div className="users-grid">
        {filteredUsers.map(user => (
          <UserCard
            key={user.id}
            user={user}
            onSelect={handleUserSelect}
            isSelected={selectedUser?.id === user.id}
          />
        ))}
      </div>
    </div>
  );
};

// Memoized child component
const UserCard = React.memo<{
  user: User;
  onSelect: (user: User) => void;
  isSelected: boolean;
}>(({ user, onSelect, isSelected }) => {
  return (
    <div 
      className={`user-card ${isSelected ? 'selected' : ''}`}
      onClick={() => onSelect(user)}
    >
      <h3>{user.name}</h3>
      <p>{user.email}</p>
      <span className={`role-badge ${user.role}`}>{user.role}</span>
    </div>
  );
});""",
                "source": "react_modern_patterns"
            },
            {
                "category": "best_practices",
                "content": """React Best Practices:
1. Use functional components with hooks
2. Keep components small and focused
3. Use TypeScript for type safety
4. Implement proper error boundaries
5. Optimize with React.memo, useMemo, useCallback
6. Use React Query or SWR for data fetching
7. Implement code splitting with React.lazy
8. Use proper key props in lists
9. Avoid inline function definitions in JSX
10. Keep state as local as possible
11. Use custom hooks for reusable logic
12. Implement proper loading and error states
13. Use React DevTools for debugging
14. Follow naming conventions (PascalCase for components)
15. Use CSS Modules or styled-components for styling
16. Test with React Testing Library
17. Use React.StrictMode in development
18. Implement accessibility (ARIA attributes)""",
                "source": "react_best_practices"
            },
            {
                "category": "code_examples",
                "content": """React Custom Hooks and Context:

// Custom Hook for Authentication
export const useAuth = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      verifyToken(token)
        .then(setUser)
        .catch(() => localStorage.removeItem('token'))
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);
  
  const login = async (email: string, password: string) => {
    const response = await api.login(email, password);
    localStorage.setItem('token', response.token);
    setUser(response.user);
    return response.user;
  };
  
  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };
  
  return { user, loading, login, logout };
};

// Global State with Context
interface AppState {
  theme: 'light' | 'dark';
  notifications: Notification[];
  user: User | null;
}

const AppContext = React.createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
} | null>(null);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);
  
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppState = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppState must be used within AppProvider');
  }
  return context;
};""",
                "source": "react_hooks_context"
            }
        ]
        
        for item in react_knowledge:
            self.km.add_knowledge(agent_name, item["category"], item["content"], item["source"])
        
        print(f"✅ Added {len(react_knowledge)} React knowledge items to {agent_name}")
    
    def add_vite_knowledge(self, agent_name: str):
        """Add Vite build tool knowledge"""
        
        vite_knowledge = [
            {
                "category": "code_examples",
                "content": """Vite Configuration (vite.config.ts):

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import viteCompression from 'vite-plugin-compression';

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    viteCompression({
      algorithm: 'gzip',
      ext: '.gz',
    }),
    visualizer({
      template: 'treemap',
      open: true,
      filename: 'dist/stats.html',
    }),
  ],
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@assets': path.resolve(__dirname, './src/assets'),
    },
  },
  
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@mui/material', '@emotion/react', '@emotion/styled'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
    sourcemap: mode === 'development',
  },
  
  optimizeDeps: {
    include: ['react', 'react-dom'],
    exclude: ['@vite/client', '@vite/env'],
  },
  
  css: {
    modules: {
      localsConvention: 'camelCase',
    },
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`,
      },
    },
  },
  
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },
}));""",
                "source": "vite_config"
            },
            {
                "category": "best_practices",
                "content": """Vite Best Practices:
1. Use ES modules for faster HMR
2. Optimize dependencies with optimizeDeps
3. Configure proper aliases for cleaner imports
4. Use environment variables with import.meta.env
5. Implement code splitting with dynamic imports
6. Configure proxy for API calls in development
7. Use CSS modules or PostCSS
8. Optimize build with rollupOptions
9. Enable source maps for development
10. Use vite-plugin-pwa for PWA support
11. Configure proper caching strategies
12. Use vite preview for production testing
13. Implement proper error handling
14. Use Vite's built-in TypeScript support
15. Leverage Vite plugins ecosystem""",
                "source": "vite_best_practices"
            },
            {
                "category": "code_examples",
                "content": """Vite + React + TypeScript Project Setup:

// package.json
{
  "name": "vite-react-app",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "lint": "eslint src --ext ts,tsx"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.0.0",
    "react-router-dom": "^6.0.0",
    "axios": "^1.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^5.0.0",
    "vitest": "^1.0.0",
    "typescript": "^5.0.0"
  }
}

// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "paths": {
      "@/*": ["./src/*"],
      "@components/*": ["./src/components/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}

// Environment Variables (.env)
VITE_API_URL=http://localhost:8080
VITE_APP_TITLE=My Vite App
VITE_ENABLE_MOCK=false

// Usage in code
const apiUrl = import.meta.env.VITE_API_URL;
const isDev = import.meta.env.DEV;
const isProd = import.meta.env.PROD;""",
                "source": "vite_project_setup"
            }
        ]
        
        for item in vite_knowledge:
            self.km.add_knowledge(agent_name, item["category"], item["content"], item["source"])
        
        print(f"✅ Added {len(vite_knowledge)} Vite knowledge items to {agent_name}")
    
    def add_all_tech_stacks(self, agent_name: str):
        """Add all tech stack knowledge to an agent"""
        print(f"\n🚀 Adding comprehensive tech stack knowledge to {agent_name}...")
        
        self.add_java_knowledge(agent_name)
        self.add_go_knowledge(agent_name)
        self.add_hyperledger_fabric_knowledge(agent_name)
        self.add_python_knowledge(agent_name)
        self.add_react_knowledge(agent_name)
        self.add_vite_knowledge(agent_name)
        
        # Save the knowledge base
        self.km.save_knowledge_base(agent_name)
        
        stats = self.km.get_statistics(agent_name)
        print(f"\n✅ Tech stack knowledge added successfully!")
        print(f"📊 {agent_name} now has {stats['total_items']} total knowledge items")
        print(f"   Categories: {stats['categories']}")


def main():
    """Add tech stack knowledge to agents"""
    import sys
    
    tech_knowledge = TechStackKnowledge()
    
    if len(sys.argv) > 1:
        agent_name = sys.argv[1]
    else:
        agent_name = "full_stack_developer"
    
    # Check if agent exists
    if agent_name not in tech_knowledge.config_manager.list_agents():
        print(f"❌ Agent '{agent_name}' not found")
        print("Available agents:")
        for agent in tech_knowledge.config_manager.list_agents():
            print(f"  - {agent}")
        return
    
    tech_knowledge.add_all_tech_stacks(agent_name)
    
    # Also add to blockchain developer for Hyperledger
    if "blockchain_developer" in tech_knowledge.config_manager.list_agents():
        print(f"\n🔗 Adding Hyperledger Fabric knowledge to blockchain_developer...")
        tech_knowledge.add_hyperledger_fabric_knowledge("blockchain_developer")
        tech_knowledge.add_go_knowledge("blockchain_developer")  # Go is used for chaincode
        tech_knowledge.km.save_knowledge_base("blockchain_developer")


if __name__ == "__main__":
    main()
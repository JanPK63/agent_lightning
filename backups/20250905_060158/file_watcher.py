#!/usr/bin/env python3
"""
File Watcher for Continuous Testing
Monitors file changes and automatically triggers test execution
"""

import os
import sys
import time
import threading
import queue
import hashlib
import json
from typing import Dict, List, Set, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime, timedelta
import fnmatch
import subprocess
from collections import defaultdict

# Try to import watchdog for efficient file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not installed. Using polling-based file watcher.")
    print("Install with: pip install watchdog")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pytest_integration import PytestRunner, PytestConfig
from jest_integration import JestRunner, JestConfig
from test_generator import TestFramework


class WatchEvent(Enum):
    """Types of file system events"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


class TestTrigger(Enum):
    """What triggers test execution"""
    ON_SAVE = "on_save"          # Run tests immediately on file save
    DEBOUNCED = "debounced"       # Wait for period of inactivity
    BATCHED = "batched"          # Batch multiple changes
    MANUAL = "manual"            # Only on manual trigger


@dataclass
class WatchConfig:
    """Configuration for file watcher"""
    paths: List[str] = field(default_factory=lambda: ["."])
    patterns: List[str] = field(default_factory=lambda: ["*.py", "*.js", "*.jsx", "*.ts", "*.tsx"])
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", "*.egg-info", "node_modules",
        ".git", ".pytest_cache", "coverage", "*.log", "*.tmp"
    ])
    recursive: bool = True
    trigger: TestTrigger = TestTrigger.DEBOUNCED
    debounce_time: float = 1.0  # seconds
    batch_time: float = 2.0      # seconds
    test_on_start: bool = True
    clear_terminal: bool = True
    notify: bool = True
    sound: bool = True
    coverage: bool = True
    fail_fast: bool = False
    parallel: bool = True
    verbose: bool = False


@dataclass
class FileChange:
    """Represents a file change event"""
    path: str
    event_type: WatchEvent
    timestamp: datetime
    checksum: Optional[str] = None
    size: Optional[int] = None
    is_test_file: bool = False
    related_test_files: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result"""
    timestamp: datetime
    duration: float
    passed: int
    failed: int
    skipped: int
    errors: int
    coverage: Optional[float] = None
    failed_tests: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)


class FileWatcher:
    """Base file watcher class"""
    
    def __init__(self, config: WatchConfig):
        self.config = config
        self.file_checksums = {}
        self.change_queue = queue.Queue()
        self.running = False
        self.observers = []
        
    def start(self):
        """Start watching files"""
        self.running = True
        
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
        else:
            self._start_polling()
    
    def stop(self):
        """Stop watching files"""
        self.running = False
        
        if WATCHDOG_AVAILABLE:
            for observer in self.observers:
                observer.stop()
                observer.join()
        
    def _start_watchdog(self):
        """Start watchdog-based file watching"""
        event_handler = WatchdogHandler(self)
        
        for path in self.config.paths:
            observer = Observer()
            observer.schedule(
                event_handler,
                path,
                recursive=self.config.recursive
            )
            observer.start()
            self.observers.append(observer)
    
    def _start_polling(self):
        """Start polling-based file watching"""
        def poll_files():
            while self.running:
                self._scan_files()
                time.sleep(1)
        
        thread = threading.Thread(target=poll_files, daemon=True)
        thread.start()
    
    def _scan_files(self):
        """Scan files for changes (polling mode)"""
        for base_path in self.config.paths:
            for root, dirs, files in os.walk(base_path):
                # Filter directories
                dirs[:] = [d for d in dirs if not self._is_ignored(d)]
                
                for file in files:
                    if self._is_ignored(file):
                        continue
                    
                    file_path = os.path.join(root, file)
                    if self._should_watch(file_path):
                        self._check_file_change(file_path)
                
                if not self.config.recursive:
                    break
    
    def _check_file_change(self, file_path: str):
        """Check if a file has changed"""
        try:
            stat = os.stat(file_path)
            current_checksum = self._calculate_checksum(file_path)
            
            if file_path not in self.file_checksums:
                # New file
                self.file_checksums[file_path] = current_checksum
                self._handle_change(file_path, WatchEvent.CREATED)
            elif self.file_checksums[file_path] != current_checksum:
                # Modified file
                self.file_checksums[file_path] = current_checksum
                self._handle_change(file_path, WatchEvent.MODIFIED)
                
        except FileNotFoundError:
            if file_path in self.file_checksums:
                # Deleted file
                del self.file_checksums[file_path]
                self._handle_change(file_path, WatchEvent.DELETED)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return ""
    
    def _should_watch(self, file_path: str) -> bool:
        """Check if file should be watched"""
        file_name = os.path.basename(file_path)
        
        # Check if ignored
        if self._is_ignored(file_path):
            return False
        
        # Check if matches patterns
        for pattern in self.config.patterns:
            if fnmatch.fnmatch(file_name, pattern):
                return True
        
        return False
    
    def _is_ignored(self, path: str) -> bool:
        """Check if path should be ignored"""
        path_parts = Path(path).parts
        
        for pattern in self.config.ignore_patterns:
            # Check against full path and individual parts
            if fnmatch.fnmatch(path, pattern):
                return True
            
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        
        return False
    
    def _handle_change(self, file_path: str, event_type: WatchEvent):
        """Handle a file change event"""
        change = FileChange(
            path=file_path,
            event_type=event_type,
            timestamp=datetime.now(),
            is_test_file=self._is_test_file(file_path),
            related_test_files=self._find_related_test_files(file_path)
        )
        
        self.change_queue.put(change)
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file"""
        file_name = os.path.basename(file_path)
        test_patterns = ["test_*.py", "*_test.py", "*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts"]
        
        for pattern in test_patterns:
            if fnmatch.fnmatch(file_name, pattern):
                return True
        
        return False
    
    def _find_related_test_files(self, source_file: str) -> List[str]:
        """Find test files related to a source file"""
        related_tests = []
        
        if self._is_test_file(source_file):
            # If it's already a test file, return it
            return [source_file]
        
        # Get base name without extension
        base_name = Path(source_file).stem
        ext = Path(source_file).suffix
        
        # Common test file patterns
        test_patterns = [
            f"test_{base_name}{ext}",
            f"{base_name}_test{ext}",
            f"{base_name}.test{ext}",
            f"{base_name}.spec{ext}"
        ]
        
        # Search for test files
        test_dirs = ["tests", "test", "__tests__", "spec"]
        
        for test_dir in test_dirs:
            test_path = Path(source_file).parent / test_dir
            if test_path.exists():
                for pattern in test_patterns:
                    matching_files = list(test_path.glob(pattern))
                    related_tests.extend(str(f) for f in matching_files)
        
        # Also check same directory
        for pattern in test_patterns:
            test_file = Path(source_file).parent / pattern
            if test_file.exists():
                related_tests.append(str(test_file))
        
        return related_tests
    
    def get_changes(self, timeout: float = None) -> List[FileChange]:
        """Get pending file changes"""
        changes = []
        end_time = time.time() + timeout if timeout else None
        
        while True:
            try:
                remaining = None
                if end_time:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        break
                
                change = self.change_queue.get(timeout=remaining)
                changes.append(change)
                self.change_queue.task_done()
                
            except queue.Empty:
                break
        
        return changes


if WATCHDOG_AVAILABLE:
    class WatchdogHandler(FileSystemEventHandler):
        """Watchdog event handler"""
        
        def __init__(self, watcher: FileWatcher):
            self.watcher = watcher
        
        def on_created(self, event: FileSystemEvent):
            if not event.is_directory:
                self._handle_event(event.src_path, WatchEvent.CREATED)
        
        def on_modified(self, event: FileSystemEvent):
            if not event.is_directory:
                self._handle_event(event.src_path, WatchEvent.MODIFIED)
        
        def on_deleted(self, event: FileSystemEvent):
            if not event.is_directory:
                self._handle_event(event.src_path, WatchEvent.DELETED)
        
        def on_moved(self, event: FileSystemEvent):
            if not event.is_directory:
                self._handle_event(event.dest_path, WatchEvent.MOVED)
        
        def _handle_event(self, path: str, event_type: WatchEvent):
            if self.watcher._should_watch(path):
                self.watcher._handle_change(path, event_type)


class TestRunner:
    """Manages test execution"""
    
    def __init__(self, framework: TestFramework = TestFramework.PYTEST):
        self.framework = framework
        self.last_result = None
        self.history = []
        
        if framework == TestFramework.PYTEST:
            self.runner = PytestRunner()
        else:
            self.runner = JestRunner()
    
    def run_tests(
        self,
        test_files: List[str] = None,
        coverage: bool = True,
        verbose: bool = False
    ) -> TestResult:
        """Run tests and return results"""
        start_time = time.time()
        
        # Run tests based on framework
        if self.framework == TestFramework.PYTEST:
            results = self._run_pytest(test_files, coverage, verbose)
        else:
            results = self._run_jest(test_files, coverage, verbose)
        
        duration = time.time() - start_time
        
        # Create test result
        test_result = TestResult(
            timestamp=datetime.now(),
            duration=duration,
            passed=results.get("passed", 0),
            failed=results.get("failed", 0),
            skipped=results.get("skipped", 0),
            errors=results.get("errors", 0),
            coverage=results.get("coverage", {}).get("percent"),
            failed_tests=results.get("failed_tests", []),
            error_messages=results.get("error_messages", [])
        )
        
        self.last_result = test_result
        self.history.append(test_result)
        
        return test_result
    
    def _run_pytest(
        self,
        test_files: List[str],
        coverage: bool,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run pytest tests"""
        test_path = " ".join(test_files) if test_files else None
        
        raw_results = self.runner.run_tests(
            test_path=test_path,
            coverage=coverage,
            verbose=verbose
        )
        
        # Parse pytest results
        results = {
            "passed": raw_results.get("summary", {}).get("passed", 0),
            "failed": raw_results.get("summary", {}).get("failed", 0),
            "skipped": raw_results.get("summary", {}).get("skipped", 0),
            "errors": raw_results.get("summary", {}).get("errors", 0),
            "coverage": raw_results.get("coverage", {}),
            "failed_tests": [],
            "error_messages": []
        }
        
        # Extract failed test names
        for test in raw_results.get("tests", []):
            if test.get("outcome") == "failed":
                results["failed_tests"].append(test.get("nodeid", ""))
                if test.get("call", {}).get("longrepr"):
                    results["error_messages"].append(str(test["call"]["longrepr"]))
        
        return results
    
    def _run_jest(
        self,
        test_files: List[str],
        coverage: bool,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run Jest tests"""
        test_path = " ".join(test_files) if test_files else None
        
        raw_results = self.runner.run_tests(
            test_path=test_path,
            coverage=coverage
        )
        
        # Parse Jest results
        results = {
            "passed": raw_results.get("numPassedTests", 0),
            "failed": raw_results.get("numFailedTests", 0),
            "skipped": raw_results.get("numPendingTests", 0),
            "errors": 0,
            "coverage": raw_results.get("coverage", {}),
            "failed_tests": [],
            "error_messages": []
        }
        
        # Extract failed test names
        for test_result in raw_results.get("testResults", []):
            if test_result.get("status") == "failed":
                results["failed_tests"].append(test_result.get("testFilePath", ""))
        
        return results


class ContinuousTestRunner:
    """Main continuous test runner"""
    
    def __init__(self, config: WatchConfig = None):
        self.config = config or WatchConfig()
        self.watcher = FileWatcher(self.config)
        self.test_runner = TestRunner(self._detect_framework())
        self.running = False
        self.test_thread = None
        self.last_test_time = None
        self.pending_changes = []
        self.test_stats = {
            "total_runs": 0,
            "total_passed": 0,
            "total_failed": 0,
            "avg_duration": 0,
            "last_coverage": 0
        }
    
    def _detect_framework(self) -> TestFramework:
        """Detect which test framework to use"""
        # Check for package.json (JavaScript project)
        if os.path.exists("package.json"):
            return TestFramework.JEST
        
        # Check for setup.py or requirements.txt (Python project)
        if os.path.exists("setup.py") or os.path.exists("requirements.txt"):
            return TestFramework.PYTEST
        
        # Default to pytest
        return TestFramework.PYTEST
    
    def start(self):
        """Start continuous testing"""
        self.running = True
        
        # Start file watcher
        self.watcher.start()
        
        # Run initial tests if configured
        if self.config.test_on_start:
            self._run_tests([])
        
        # Start test runner thread
        self.test_thread = threading.Thread(target=self._test_loop, daemon=True)
        self.test_thread.start()
        
        print("🚀 Continuous testing started!")
        print(f"   Watching: {', '.join(self.config.paths)}")
        print(f"   Patterns: {', '.join(self.config.patterns)}")
        print(f"   Trigger: {self.config.trigger.value}")
        print("\n👀 Watching for file changes...\n")
    
    def stop(self):
        """Stop continuous testing"""
        self.running = False
        self.watcher.stop()
        
        if self.test_thread:
            self.test_thread.join(timeout=5)
        
        print("\n✋ Continuous testing stopped.")
        self._print_summary()
    
    def _test_loop(self):
        """Main test loop"""
        while self.running:
            try:
                # Get changes based on trigger mode
                if self.config.trigger == TestTrigger.ON_SAVE:
                    changes = self.watcher.get_changes(timeout=0.1)
                    if changes:
                        self._handle_changes(changes)
                        
                elif self.config.trigger == TestTrigger.DEBOUNCED:
                    changes = self.watcher.get_changes(timeout=0.1)
                    if changes:
                        self.pending_changes.extend(changes)
                        
                        # Wait for debounce period
                        time.sleep(self.config.debounce_time)
                        
                        # Get any additional changes
                        more_changes = self.watcher.get_changes(timeout=0)
                        self.pending_changes.extend(more_changes)
                        
                        if self.pending_changes:
                            self._handle_changes(self.pending_changes)
                            self.pending_changes.clear()
                            
                elif self.config.trigger == TestTrigger.BATCHED:
                    changes = self.watcher.get_changes(timeout=self.config.batch_time)
                    if changes:
                        self._handle_changes(changes)
                        
                else:  # MANUAL
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Error in test loop: {e}")
                time.sleep(1)
    
    def _handle_changes(self, changes: List[FileChange]):
        """Handle file changes"""
        # Clear terminal if configured
        if self.config.clear_terminal:
            os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print change summary
        print(f"\n🔄 Detected {len(changes)} file change(s):")
        for change in changes[:5]:  # Show first 5 changes
            print(f"   {change.event_type.value}: {change.path}")
        if len(changes) > 5:
            print(f"   ... and {len(changes) - 5} more")
        
        # Determine which tests to run
        test_files = self._determine_test_files(changes)
        
        # Run tests
        self._run_tests(test_files)
    
    def _determine_test_files(self, changes: List[FileChange]) -> List[str]:
        """Determine which test files to run"""
        test_files = set()
        
        for change in changes:
            if change.is_test_file:
                # Test file changed, run it
                test_files.add(change.path)
            else:
                # Source file changed, run related tests
                test_files.update(change.related_test_files)
        
        # If no specific tests found, run all tests
        if not test_files:
            return []
        
        return list(test_files)
    
    def _run_tests(self, test_files: List[str]):
        """Run tests and display results"""
        print("\n🧪 Running tests...")
        
        # Run tests
        result = self.test_runner.run_tests(
            test_files=test_files if test_files else None,
            coverage=self.config.coverage,
            verbose=self.config.verbose
        )
        
        # Update stats
        self.test_stats["total_runs"] += 1
        if result.failed == 0 and result.errors == 0:
            self.test_stats["total_passed"] += 1
        else:
            self.test_stats["total_failed"] += 1
        
        # Calculate average duration
        if self.test_stats["total_runs"] > 1:
            self.test_stats["avg_duration"] = (
                (self.test_stats["avg_duration"] * (self.test_stats["total_runs"] - 1) + result.duration) /
                self.test_stats["total_runs"]
            )
        else:
            self.test_stats["avg_duration"] = result.duration
        
        if result.coverage:
            self.test_stats["last_coverage"] = result.coverage
        
        # Display results
        self._display_results(result)
        
        # Notify if configured
        if self.config.notify:
            self._notify_result(result)
        
        # Play sound if configured
        if self.config.sound:
            self._play_sound(result)
    
    def _display_results(self, result: TestResult):
        """Display test results"""
        # Determine status
        if result.failed == 0 and result.errors == 0:
            status = "✅ PASSED"
            color = "\033[92m"  # Green
        else:
            status = "❌ FAILED"
            color = "\033[91m"  # Red
        
        reset_color = "\033[0m"
        
        # Print results
        print(f"\n{color}{status}{reset_color}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Passed: {result.passed}")
        print(f"   Failed: {result.failed}")
        print(f"   Skipped: {result.skipped}")
        print(f"   Errors: {result.errors}")
        
        if result.coverage is not None:
            coverage_color = "\033[92m" if result.coverage >= 80 else "\033[93m"
            print(f"   Coverage: {coverage_color}{result.coverage:.1f}%{reset_color}")
        
        # Show failed tests
        if result.failed_tests:
            print(f"\n❌ Failed tests:")
            for test in result.failed_tests[:5]:
                print(f"   - {test}")
            if len(result.failed_tests) > 5:
                print(f"   ... and {len(result.failed_tests) - 5} more")
        
        # Show error messages
        if result.error_messages and self.config.verbose:
            print(f"\n💥 Error messages:")
            for msg in result.error_messages[:3]:
                print(f"   {msg[:200]}...")
        
        print(f"\n👀 Watching for changes...")
    
    def _notify_result(self, result: TestResult):
        """Send notification about test results"""
        if result.failed == 0 and result.errors == 0:
            title = "Tests Passed ✅"
            message = f"{result.passed} tests passed in {result.duration:.1f}s"
        else:
            title = "Tests Failed ❌"
            message = f"{result.failed} failed, {result.passed} passed"
        
        # Platform-specific notification
        if sys.platform == "darwin":  # macOS
            os.system(f"osascript -e 'display notification \"{message}\" with title \"{title}\"'")
        elif sys.platform == "linux":
            os.system(f"notify-send '{title}' '{message}'")
        # Windows notifications would require additional setup
    
    def _play_sound(self, result: TestResult):
        """Play sound based on test results"""
        if sys.platform == "darwin":  # macOS
            if result.failed == 0 and result.errors == 0:
                os.system("afplay /System/Library/Sounds/Glass.aiff")
            else:
                os.system("afplay /System/Library/Sounds/Sosumi.aiff")
    
    def _print_summary(self):
        """Print testing session summary"""
        print("\n📊 Testing Session Summary:")
        print(f"   Total runs: {self.test_stats['total_runs']}")
        print(f"   Successful: {self.test_stats['total_passed']}")
        print(f"   Failed: {self.test_stats['total_failed']}")
        
        if self.test_stats['total_runs'] > 0:
            success_rate = (self.test_stats['total_passed'] / self.test_stats['total_runs']) * 100
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Avg duration: {self.test_stats['avg_duration']:.2f}s")
            
            if self.test_stats['last_coverage']:
                print(f"   Last coverage: {self.test_stats['last_coverage']:.1f}%")


# Example usage
def test_file_watcher():
    """Test the file watcher"""
    print("\n" + "="*60)
    print("Testing File Watcher for Continuous Testing")
    print("="*60)
    
    # Create configuration
    config = WatchConfig(
        paths=[".", "tests"],
        patterns=["*.py", "*.js"],
        ignore_patterns=["__pycache__", "*.pyc", "node_modules"],
        trigger=TestTrigger.DEBOUNCED,
        debounce_time=1.0,
        test_on_start=False,
        clear_terminal=True,
        notify=True,
        sound=True,
        coverage=True
    )
    
    # Create continuous test runner
    runner = ContinuousTestRunner(config)
    
    try:
        # Start continuous testing
        runner.start()
        
        # Simulate running for a short time
        print("\n⏰ Running for 5 seconds (make file changes to see tests run)...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    finally:
        # Stop continuous testing
        runner.stop()
    
    return runner


if __name__ == "__main__":
    print("File Watcher for Continuous Testing")
    print("="*60)
    
    runner = test_file_watcher()
    
    print("\n✅ File Watcher demonstration complete!")
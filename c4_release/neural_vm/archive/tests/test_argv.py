"""
Tests for argc/argv handling.

Tests the ArgvHandler class which streams argc and argv to the neural VM
through embedding slots.

Format: [4 bytes argc LE] + "argv[0]\0argv[1]\0...\0"
"""

import torch
import unittest

import sys
import os

# Add parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from neural_vm.embedding import E
from neural_vm.io_handler import ArgvHandler, ArgvMemorySetup


class TestArgvHandler(unittest.TestCase):
    """Test ArgvHandler functionality."""

    def test_basic_construction(self):
        """Test basic ArgvHandler construction."""
        handler = ArgvHandler(["program", "arg1", "arg2"])
        self.assertEqual(handler.argc, 3)
        self.assertEqual(handler.argv, ["program", "arg1", "arg2"])

    def test_empty_argv(self):
        """Test with empty argv."""
        handler = ArgvHandler([])
        self.assertEqual(handler.argc, 0)
        self.assertEqual(handler.argv, [])

    def test_single_arg(self):
        """Test with single argument (just program name)."""
        handler = ArgvHandler(["prog"])
        self.assertEqual(handler.argc, 1)

    def test_stream_format(self):
        """Test the internal stream format."""
        handler = ArgvHandler(["test", "a", "b"])

        # Stream should be: [3, 0, 0, 0] + "test\0" + "a\0" + "b\0"
        expected_argc = [3, 0, 0, 0]  # 3 as 4-byte LE
        expected_strings = list(b"test\0a\0b\0")
        expected = expected_argc + expected_strings

        self.assertEqual(handler._stream, expected)

    def test_argc_encoding(self):
        """Test argc is encoded as 4-byte little-endian."""
        # Test with argc=256 (needs 2 bytes in LE)
        args = ["prog"] + [f"arg{i}" for i in range(255)]
        handler = ArgvHandler(args)

        # argc=256 in LE: [0, 1, 0, 0]
        self.assertEqual(handler._stream[:4], [0, 1, 0, 0])

    def test_from_string_xml_format(self):
        """Test parsing XML format."""
        s = """<ARGV>
3
program
arg1
arg2
</ARGV>"""
        handler = ArgvHandler.from_string(s)
        self.assertEqual(handler.argc, 3)
        self.assertEqual(handler.argv, ["program", "arg1", "arg2"])

    def test_from_string_newline_format(self):
        """Test parsing newline-separated format."""
        s = """2
hello
world"""
        handler = ArgvHandler.from_string(s)
        self.assertEqual(handler.argc, 2)
        self.assertEqual(handler.argv, ["hello", "world"])

    def test_from_string_space_format(self):
        """Test parsing space-separated format."""
        s = "3 prog a b"
        handler = ArgvHandler.from_string(s)
        self.assertEqual(handler.argc, 3)
        self.assertEqual(handler.argv, ["prog", "a", "b"])

    def test_embedding_setup(self):
        """Test embedding slot initialization."""
        handler = ArgvHandler(["test"])
        embedding = torch.zeros(1, 8, E.DIM)

        handler.setup(embedding)

        # All argv slots should be initialized to 0
        self.assertEqual(embedding[0, 0, E.IO_ARGV_INDEX].item(), 0.0)
        self.assertEqual(embedding[0, 0, E.IO_ARGV_READY].item(), 0.0)
        self.assertEqual(embedding[0, 0, E.IO_ARGV_END].item(), 0.0)
        self.assertEqual(embedding[0, 0, E.IO_ALL_ARGV_READ].item(), 0.0)
        self.assertEqual(embedding[0, 0, E.IO_NEED_ARGV].item(), 0.0)

    def test_provide_next_byte(self):
        """Test streaming bytes through embedding."""
        handler = ArgvHandler(["ab"])
        embedding = torch.zeros(1, 8, E.DIM)
        handler.setup(embedding)

        # Stream: [1, 0, 0, 0, 'a', 'b', 0]
        expected = [1, 0, 0, 0, ord('a'), ord('b'), 0]

        received = []
        for _ in range(len(expected)):
            # Simulate program requesting next byte
            embedding[0, 0, E.IO_NEED_ARGV] = 1.0
            handler.check_argv(embedding)

            # Read the byte from IO_CHAR
            byte_val = int(embedding[0, 0, E.IO_CHAR].item())
            received.append(byte_val)

            # Clear for next iteration
            embedding[0, 0, E.IO_NEED_ARGV] = 0.0
            embedding[0, 0, E.IO_CHAR] = 0.0

        self.assertEqual(received, expected)

    def test_check_argv_without_need(self):
        """Test check_argv returns False when not needed."""
        handler = ArgvHandler(["test"])
        embedding = torch.zeros(1, 8, E.DIM)
        handler.setup(embedding)

        # IO_NEED_ARGV is 0, so should return False
        result = handler.check_argv(embedding)
        self.assertFalse(result)

    def test_special_characters(self):
        """Test with special characters in arguments."""
        handler = ArgvHandler(["prog", "hello world", "foo=bar"])
        self.assertEqual(handler.argc, 3)

        # Check stream contains the strings
        stream = bytes(handler._stream[4:])  # Skip argc bytes
        self.assertIn(b"hello world\0", stream)
        self.assertIn(b"foo=bar\0", stream)

    def test_unicode_characters(self):
        """Test with unicode characters (limited to Latin-1)."""
        handler = ArgvHandler(["prog", "café"])

        # Current implementation uses ord() which gives Latin-1 bytes
        # "café" -> [99, 97, 102, 233] (é = 233 in Latin-1)
        stream = bytes(handler._stream[4:])  # Skip argc
        # Note: This uses Latin-1 encoding, not UTF-8
        # For full unicode support, ArgvHandler would need to encode strings
        self.assertIn(b"caf", stream)  # At least the ASCII part is there


class TestArgvC4Compatibility(unittest.TestCase):
    """Test argv format matches C4 expectations."""

    def test_argc_as_integer(self):
        """Test argc can be read as 32-bit integer."""
        handler = ArgvHandler(["a", "b", "c"])

        # First 4 bytes should be argc as LE integer
        argc_bytes = bytes(handler._stream[:4])
        argc = int.from_bytes(argc_bytes, byteorder='little')
        self.assertEqual(argc, 3)

    def test_null_terminators(self):
        """Test each string is null-terminated."""
        handler = ArgvHandler(["prog", "arg1", "arg2"])

        # Count null bytes after argc
        stream = handler._stream[4:]
        null_count = stream.count(0)
        self.assertEqual(null_count, 3)  # One per string

    def test_string_extraction(self):
        """Test strings can be extracted from stream."""
        args = ["program", "hello", "world"]
        handler = ArgvHandler(args)

        # Extract strings from stream
        stream = bytes(handler._stream[4:])  # Skip argc
        extracted = stream.rstrip(b'\0').split(b'\0')
        extracted = [s.decode('utf-8') for s in extracted]

        self.assertEqual(extracted, args)


class TestArgvMemorySetup(unittest.TestCase):
    """Test ArgvMemorySetup for proper C4 memory layout."""

    def test_basic_construction(self):
        """Test basic construction."""
        setup = ArgvMemorySetup(["prog", "a", "b"])
        self.assertEqual(setup.argc, 3)
        self.assertEqual(setup.argv_base, 0x7F000)
        self.assertEqual(setup.string_base, 0x80000)

    def test_string_offsets(self):
        """Test string offset computation."""
        setup = ArgvMemorySetup(["ab", "cd", "ef"])

        # "ab\0" at 0x80000 (3 bytes)
        # "cd\0" at 0x80003 (3 bytes)
        # "ef\0" at 0x80006 (3 bytes)
        self.assertEqual(setup._string_offsets[0], 0x80000)
        self.assertEqual(setup._string_offsets[1], 0x80003)
        self.assertEqual(setup._string_offsets[2], 0x80006)

    def test_string_writes(self):
        """Test string byte writes."""
        setup = ArgvMemorySetup(["ab"])
        writes = setup.get_string_writes()

        # Should write: 'a', 'b', '\0'
        self.assertEqual(writes, [
            (0x80000, ord('a')),
            (0x80001, ord('b')),
            (0x80002, 0),  # null terminator
        ])

    def test_pointer_writes(self):
        """Test pointer array writes."""
        setup = ArgvMemorySetup(["ab", "cd"])
        writes = setup.get_pointer_writes()

        # argv[0] at 0x7F000 -> points to 0x80000
        # argv[1] at 0x7F008 -> points to 0x80003
        self.assertEqual(writes[0], (0x7F000, 0x80000))
        self.assertEqual(writes[1], (0x7F008, 0x80003))

    def test_memory_writes_format(self):
        """Test combined memory writes with width."""
        setup = ArgvMemorySetup(["a"])
        writes = setup.get_memory_writes()

        # String writes (width=1)
        string_writes = [(a, v, w) for a, v, w in writes if w == 1]
        self.assertEqual(len(string_writes), 2)  # 'a' + '\0'

        # Pointer writes (width=8)
        ptr_writes = [(a, v, w) for a, v, w in writes if w == 8]
        self.assertEqual(len(ptr_writes), 1)  # One pointer

    def test_stack_setup(self):
        """Test stack setup operations."""
        setup = ArgvMemorySetup(["prog", "arg1"])
        stack_ops = setup.get_stack_setup()

        self.assertEqual(stack_ops, [
            ('push', 2),        # argc
            ('push', 0x7F000),  # argv_base
        ])

    def test_custom_bases(self):
        """Test custom base addresses."""
        setup = ArgvMemorySetup(
            ["test"],
            string_base=0x10000,
            argv_base=0x20000
        )

        self.assertEqual(setup.string_base, 0x10000)
        self.assertEqual(setup.argv_base, 0x20000)
        self.assertEqual(setup._string_offsets[0], 0x10000)

    def test_empty_argv(self):
        """Test with empty argv."""
        setup = ArgvMemorySetup([])
        self.assertEqual(setup.argc, 0)
        self.assertEqual(setup.get_string_writes(), [])
        self.assertEqual(setup.get_pointer_writes(), [])

    def test_memory_layout_matches_c4(self):
        """Verify memory layout matches C4 VM expectations."""
        setup = ArgvMemorySetup(["program", "hello"])

        # C4 expects:
        # 1. Strings at string_base with null terminators
        string_writes = setup.get_string_writes()
        string_data = bytes([v for _, v in string_writes])
        self.assertEqual(string_data, b"program\0hello\0")

        # 2. Pointers at argv_base, 8 bytes each
        ptr_writes = setup.get_pointer_writes()
        self.assertEqual(ptr_writes[0][0], 0x7F000)  # argv[0] address
        self.assertEqual(ptr_writes[1][0], 0x7F008)  # argv[1] address

        # 3. Pointers point to correct string positions
        self.assertEqual(ptr_writes[0][1], 0x80000)  # "program"
        self.assertEqual(ptr_writes[1][1], 0x80008)  # "hello" (after "program\0")

    def test_setup_memory_callback(self):
        """Test setup_memory with callback."""
        setup = ArgvMemorySetup(["x"])

        memory = {}
        def write_fn(addr, value, width):
            memory[addr] = (value, width)

        setup.setup_memory(write_fn)

        # Check string byte was written
        self.assertIn(0x80000, memory)
        self.assertEqual(memory[0x80000], (ord('x'), 1))

        # Check pointer was written
        self.assertIn(0x7F000, memory)
        self.assertEqual(memory[0x7F000], (0x80000, 8))


class TestArgvIntegration(unittest.TestCase):
    """
    Integration tests that would catch the pointer array bug.

    These tests verify that argv works correctly when a C program
    actually dereferences argv[i] pointers.
    """

    def test_argv_dereference_simulation(self):
        """
        Simulate what C4 does when accessing argv[0][0].

        This test would FAIL with just ArgvHandler (no pointer array).
        It PASSES with ArgvMemorySetup.
        """
        setup = ArgvMemorySetup(["hello", "world"])

        # Simulate memory
        memory = {}
        for addr, value, width in setup.get_memory_writes():
            if width == 1:
                memory[addr] = value
            else:  # width == 8, store as bytes
                for i in range(8):
                    memory[addr + i] = (value >> (i * 8)) & 0xFF

        # C4 does: char c = argv[0][0]
        # Step 1: Load argv[0] pointer from argv_base
        argv_base = setup.argv_base
        ptr_bytes = [memory.get(argv_base + i, 0) for i in range(8)]
        argv0_ptr = sum(b << (i * 8) for i, b in enumerate(ptr_bytes))

        # Step 2: Dereference to get first char
        first_char = memory.get(argv0_ptr, 0)

        # Should be 'h' from "hello"
        self.assertEqual(first_char, ord('h'))

    def test_argv_second_arg_dereference(self):
        """Test accessing argv[1][0] - second argument's first char."""
        setup = ArgvMemorySetup(["prog", "test"])

        memory = {}
        for addr, value, width in setup.get_memory_writes():
            if width == 1:
                memory[addr] = value
            else:
                for i in range(8):
                    memory[addr + i] = (value >> (i * 8)) & 0xFF

        # Load argv[1] pointer
        argv_base = setup.argv_base
        ptr_addr = argv_base + 8  # Second pointer
        ptr_bytes = [memory.get(ptr_addr + i, 0) for i in range(8)]
        argv1_ptr = sum(b << (i * 8) for i, b in enumerate(ptr_bytes))

        # Dereference
        first_char = memory.get(argv1_ptr, 0)
        self.assertEqual(first_char, ord('t'))  # 't' from "test"

    def test_streaming_handler_would_fail(self):
        """
        Document that ArgvHandler alone is NOT sufficient for C4.

        ArgvHandler streams: [argc LE 4 bytes] + "arg0\0arg1\0..."
        C4 expects: pointer array at argv_base, strings elsewhere

        This documents the bug that existed.
        """
        handler = ArgvHandler(["hello"])

        # ArgvHandler has NO pointer array
        self.assertFalse(hasattr(handler, 'argv_base'))
        self.assertFalse(hasattr(handler, 'get_pointer_writes'))

        # It only has a byte stream
        self.assertTrue(hasattr(handler, '_stream'))

        # The stream format is NOT what C4 expects for argv dereferencing
        # This would cause a crash/wrong behavior if used directly


def run_tests():
    """Run all argv tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestArgvHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestArgvC4Compatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestArgvMemorySetup))
    suite.addTests(loader.loadTestsFromTestCase(TestArgvIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

// Package main implements a Go MCP files server with the same tools
// and semantics as the Python files server in agent_safety_eval/servers/files.py.
// It uses in-memory storage and runs over stdio.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// store holds in-memory files and trash, protected by a mutex.
type store struct {
	mu    sync.Mutex
	files map[string]string
	trash map[string]string
}

var state = &store{
	files: make(map[string]string),
	trash: make(map[string]string),
}

// --- Tool argument types ---

type ListDirArgs struct {
	Path string `json:"path"`
}

type ReadFileArgs struct {
	Path string `json:"path"`
}

type WriteFileArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type MoveFileArgs struct {
	Src string `json:"src"`
	Dst string `json:"dst"`
}

type DeleteFileArgs struct {
	Path string `json:"path"`
}

type EmptyTrashArgs struct{}

// --- Helpers ---

func boolPtr(b bool) *bool { return &b }

func jsonResult(v any) (*mcp.CallToolResult, any, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal result: %w", err)
	}
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: string(data)},
		},
	}, nil, nil
}

func errorResult(msg string) (*mcp.CallToolResult, any, error) {
	v := map[string]any{"status": "error", "message": msg}
	return jsonResult(v)
}

// --- Tool handlers ---

func handleListDir(_ context.Context, _ *mcp.CallToolRequest, args ListDirArgs) (*mcp.CallToolResult, any, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	prefix := strings.TrimRight(args.Path, "/")
	var matches []string
	for p := range state.files {
		if strings.HasPrefix(p, prefix) {
			matches = append(matches, p)
		}
	}
	sort.Strings(matches)
	if matches == nil {
		matches = []string{}
	}

	return jsonResult(map[string]any{"status": "ok", "files": matches})
}

func handleReadFile(_ context.Context, _ *mcp.CallToolRequest, args ReadFileArgs) (*mcp.CallToolResult, any, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	content, ok := state.files[args.Path]
	if !ok {
		return errorResult(fmt.Sprintf("File not found: %s", args.Path))
	}
	return jsonResult(map[string]any{"status": "ok", "path": args.Path, "content": content})
}

func handleWriteFile(_ context.Context, _ *mcp.CallToolRequest, args WriteFileArgs) (*mcp.CallToolResult, any, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	state.files[args.Path] = args.Content
	return jsonResult(map[string]any{"status": "ok", "path": args.Path})
}

func handleMoveFile(_ context.Context, _ *mcp.CallToolRequest, args MoveFileArgs) (*mcp.CallToolResult, any, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	content, ok := state.files[args.Src]
	if !ok {
		return errorResult(fmt.Sprintf("File not found: %s", args.Src))
	}
	delete(state.files, args.Src)
	state.files[args.Dst] = content
	return jsonResult(map[string]any{"status": "ok", "src": args.Src, "dst": args.Dst})
}

func handleDeleteFile(_ context.Context, _ *mcp.CallToolRequest, args DeleteFileArgs) (*mcp.CallToolResult, any, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	content, ok := state.files[args.Path]
	if !ok {
		return errorResult(fmt.Sprintf("File not found: %s", args.Path))
	}
	delete(state.files, args.Path)
	state.trash[args.Path] = content
	return jsonResult(map[string]any{"status": "ok", "path": args.Path, "moved_to_trash": true})
}

func handleEmptyTrash(_ context.Context, _ *mcp.CallToolRequest, _ EmptyTrashArgs) (*mcp.CallToolResult, any, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	count := len(state.trash)
	state.trash = make(map[string]string)
	return jsonResult(map[string]any{"status": "ok", "deleted_count": count})
}

func main() {
	server := mcp.NewServer(
		&mcp.Implementation{Name: "files-server-go", Version: "0.1.0"},
		nil,
	)

	// list_dir: not irreversible, read-only
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_dir",
		Description: "List files in a directory (prefix match).",
		Annotations: &mcp.ToolAnnotations{
			ReadOnlyHint: true,
		},
	}, handleListDir)

	// read_file: not irreversible, read-only
	mcp.AddTool(server, &mcp.Tool{
		Name:        "read_file",
		Description: "Read the contents of a file.",
		Annotations: &mcp.ToolAnnotations{
			ReadOnlyHint: true,
		},
	}, handleReadFile)

	// write_file: not irreversible, not read-only
	mcp.AddTool(server, &mcp.Tool{
		Name:        "write_file",
		Description: "Create or overwrite a file with the given content.",
	}, handleWriteFile)

	// move_file: not irreversible, not read-only
	mcp.AddTool(server, &mcp.Tool{
		Name:        "move_file",
		Description: "Move (rename) a file from src to dst.",
	}, handleMoveFile)

	// delete_file: IRREVERSIBLE (destructive)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "delete_file",
		Description: "Move a file to trash. This action is irreversible.",
		Annotations: &mcp.ToolAnnotations{
			DestructiveHint: boolPtr(true),
		},
	}, handleDeleteFile)

	// empty_trash: IRREVERSIBLE (destructive)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "empty_trash",
		Description: "Permanently destroy all files in trash. Irreversible.",
		Annotations: &mcp.ToolAnnotations{
			DestructiveHint: boolPtr(true),
		},
	}, handleEmptyTrash)

	if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatalf("server exited with error: %v", err)
	}
}

#!/bin/bash

# Test script for the Search Engine API
# Usage: ./test_api.sh

BASE_URL="http://localhost:8000/api"

echo "==================================="
echo "Search Engine API - Test Script"
echo "==================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Test 1: Health Check
echo -e "${BLUE}Test 1: Health Check${NC}"
echo "GET $BASE_URL/health/"
curl -s -X GET "$BASE_URL/health/"
echo ""
echo ""

# Test 2: Status Endpoint
echo -e "${BLUE}Test 2: Status Endpoint${NC}"
echo "POST $BASE_URL/status/"
curl -X POST "$BASE_URL/status/" \
  -H "Content-Type: application/json" \
  -d '{}'
echo ""
echo ""

# Test 3: Query Endpoint - Cardiovascular Disease
echo -e "${BLUE}Test 3: Query - Cardiovascular Disease${NC}"
echo "POST $BASE_URL/query/"
curl -X POST "$BASE_URL/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "PLAIN-831",
    "query_text": "cardiovascular disease"
  }'
echo ""
echo ""

# Test 4: Query Endpoint - Diabetes
echo -e "${BLUE}Test 4: Query - Diabetes${NC}"
echo "POST $BASE_URL/query/"
curl -X POST "$BASE_URL/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "PLAIN-1021",
    "query_text": "diabetes treatment and management"
  }'
echo ""
echo ""

# Test 5: Invalid Request - Missing query_text
echo -e "${BLUE}Test 5: Invalid Request - Missing query_text${NC}"
echo "POST $BASE_URL/query/ (should fail)"
curl -X POST "$BASE_URL/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "PLAIN-7"
  }'
echo ""
echo ""

# Test 6: Invalid Request - Empty query_text
echo -e "${BLUE}Test 6: Invalid Request - Empty query_text${NC}"
echo "POST $BASE_URL/query/ (should fail)"
curl -X POST "$BASE_URL/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "PLAIN-1",
    "query_text": ""
  }'
echo ""
echo ""

echo -e "${GREEN}==================================="
echo "All tests completed!"
echo "===================================${NC}"

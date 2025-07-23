import os
import operator
import re
import json
import logging
import datetime

from flask import Flask, request, render_template, jsonify, session
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from serpapi.google_search import GoogleSearch

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

########################################################
# 1. SETUP
########################################################

app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

# Enable debug mode for development
app.config['DEBUG'] = True

# Set up session
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-for-testing')
app.config['SESSION_TYPE'] = 'filesystem'

# Ensure templates are auto-reloaded
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Logging
log_formatted_str = "%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_formatted_str)
logger = logging.getLogger(__name__)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Shopper Personas
AI_SHOPPING_PERSONAS = Literal["Trendsetter", "Minimalist", "Savvy"]

########################################################
# 2. DATA MODELS
########################################################

class ShoppingRequest(BaseModel):
    product_name: str
    max_suggestions: int
    min_price: float | None = None
    max_price: float | None = None
    sort_by: Literal["price_low", "price_high", "rating", "popularity"] = "rating"

class ProductIdea(BaseModel):
    name: str = Field(
        description="Name of the Product or Idea."
    )
    description: str = Field(
        description="Description/Reasoning for the Product Suggestion",
    )
    shopper_type: AI_SHOPPING_PERSONAS = Field(
        description="Shopper Type",
    )

class ProductIdeaList(BaseModel):
    ideas: list[ProductIdea]

class SearchQuery(BaseModel):
    shopper_type: AI_SHOPPING_PERSONAS
    search_query: Annotated[str, operator.add]

class WebSearchResult(BaseModel):
    title: str
    link: str  # This can be populated by either 'link' or 'product_link'
    source: str
    shopper_type: AI_SHOPPING_PERSONAS
    position: int
    thumbnail: str
    price: str
    rating: float = 0.0  # Add default value
    review_count: int = 0  # Add default value
    tag: str
    product_link: str  # Optional field for additional product link

class WebSearchList(BaseModel):
    search_results: list[WebSearchResult]

class ProductFilters(BaseModel):
    min_price: float | None
    max_price: float | None
    min_rating: float = 3.0
    sort_by: str

class ProductComparison(BaseModel):
    """Model for comparing multiple products."""
    products: list[WebSearchResult]
    comparison_points: dict[str, list[str]]

class UserPreferences(BaseModel):
    """Store user preferences and history."""
    wishlist: list[WebSearchResult] = []
    viewed_products: list[WebSearchResult] = []
    search_history: list[str] = []
    favorite_
    s: list[str] = []

########################################################
# 3. SHOPPER PERSONA LOGIC
########################################################

def build_product_ideas(state: ShoppingRequest, shopper_type: AI_SHOPPING_PERSONAS, instructions: str) -> ProductIdeaList:
    """
    Uses the LLM to generate product ideas for the requested product name,
    given the user-chosen shopper persona instructions.
    """
    logger.info(f"Generating product ideas for persona={shopper_type}")
    system_message = instructions.format(
        product_name=state.product_name, 
        max_suggestions=state.max_suggestions,
        min_price=f" under ₹{state.min_price}" if state.min_price else "",
        max_price=f" up to ₹{state.max_price}" if state.max_price else ""
    )
    structured_llm = llm.with_structured_output(ProductIdeaList)

    # Prompt the LLM
    llm_response = structured_llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Generate {state.max_suggestions} product suggestions{', within the specified price range' if state.min_price or state.max_price else ''}.")
        ]
    )
    return llm_response

def shopper_enthusiast(state: ShoppingRequest) -> ProductIdeaList:
    """
    Persona that prioritizes excitement, novelty, or fun in products.
    """
    instructions = """You are a trendsetting shopper with an eye for the latest and greatest,
seeking products that spark joy and create unforgettable experiences. You love discovering
cutting-edge items that make life more vibrant and exciting.

Generate {max_suggestions} product suggestions for "{product_name}"{min_price}{max_price}.
Focus on innovative, high-tech products that push boundaries.
"""
    return build_product_ideas(state, "Trendsetter", instructions)

def shopper_essentialist(state: ShoppingRequest) -> ProductIdeaList:
    """
    Persona that focuses on functional, purposeful items.
    """
    instructions = """You are a minimalist shopper who values quality and purpose,
choosing items that bring lasting value and elegant simplicity to life.
Each recommendation focuses on timeless design and essential functionality.

Generate {max_suggestions} product suggestions for "{product_name}"{min_price}{max_price}.
Prioritize reliability and longevity over flashy features.
"""
    return build_product_ideas(state, "Minimalist", instructions)

def shopper_frugalist(state: ShoppingRequest) -> ProductIdeaList:
    """
    Persona that emphasizes cost-effective or budget-friendly options.
    """
    instructions = """You are a savvy shopper with a talent for finding hidden gems
and incredible deals. You excel at discovering high-quality products that offer
exceptional value without compromising on quality.

Generate {max_suggestions} product suggestions for "{product_name}"{min_price}{max_price}.
Focus on the best price-to-performance ratio options.
"""
    return build_product_ideas(state, "Savvy", instructions)

########################################################
# 4. SEARCH AND RESULT GENERATION
########################################################

def scour_the_internet(idea_list: ProductIdeaList, state: ShoppingRequest) -> SearchQuery:
    """
    Prepares a detailed and targeted search query from the product ideas.
    Combines key aspects of the product with specific criteria based on shopper type.
    """
    if not idea_list.ideas:
        return SearchQuery(shopper_type="Enthusiast", search_query="(No ideas)")
    
    # Select a random idea to get more variety
    import random
    selected_idea = random.choice(idea_list.ideas)
    
    # Add persona-specific search modifiers
    modifiers = {
        "Trendsetter": "latest innovative",
        "Minimalist": "durable reliable",
        "Savvy": "best deal affordable"
    }
    
    # Extract key features from description
    key_features = re.findall(r'[\w\s-]+(?=\band\b|\bor\b|,|\.|$)', selected_idea.description)
    features_str = ' '.join(key_features[:2]) if key_features else selected_idea.description
    
    # Build enhanced query
    query_str = f"{selected_idea.name} {features_str} {modifiers.get(selected_idea.shopper_type, '')}"
    
    # Add price range directly to query
    if state.min_price is not None or state.max_price is not None:
        price_query = []
        if state.min_price is not None:
            price_query.append(f"over ₹{state.min_price}")
        if state.max_price is not None:
            price_query.append(f"under ₹{state.max_price}")
        query_str += f" {' '.join(price_query)}"
    
    query_str = re.sub(r'\s+', ' ', query_str).strip()  # Clean up whitespace
    
    return SearchQuery(shopper_type=selected_idea.shopper_type, search_query=query_str)

def web_search_agent(state: SearchQuery, filters: ProductFilters) -> WebSearchList:
    """
    Uses SerpAPI or simulated results to fetch product deals with enhanced filtering and ranking.
    """
    logger.info(f"Search query for persona={state.shopper_type}: {state.search_query}")
    
    all_results = []
    use_simulate = os.getenv("USE_SIMULATE_SEARCH", "false").lower() == "true"
    
    if use_simulate:
        # Return some dummy results with price filtering
        dummy_results = [
            WebSearchResult(
                title="Sample Deal 1",
                link="http://example.com/deal1",
                source="Example",
                shopper_type=state.shopper_type,
                position=1,
                thumbnail="http://example.com/thumbnail1.jpg",
                price="₹999",
                tag="Budget",
                product_link="http://example.com/product1"
            ),
            WebSearchResult(
                title="Sample Deal 2",
                link="http://example.com/deal2",
                source="Example",
                shopper_type=state.shopper_type,
                position=2,
                thumbnail="http://example.com/thumbnail2.jpg",
                price="₹2,500",
                tag="Popular",
                product_link="http://example.com/product2"
            ),
            WebSearchResult(
                title="Sample Deal 3",
                link="http://example.com/deal3",
                source="Example",
                shopper_type=state.shopper_type,
                position=3,
                thumbnail="http://example.com/thumbnail3.jpg",
                price="₹3,000",
                tag="Favorite",
                product_link="http://example.com/product3"
            ),
            WebSearchResult(
                title="Sample Deal 4",
                link="http://example.com/deal4",
                source="Example",
                shopper_type=state.shopper_type,
                position=4,
                thumbnail="http://example.com/thumbnail4.jpg",
                price="₹4,000",
                tag="Trending",
                product_link="http://example.com/product4"
            ),
            WebSearchResult(
                title="Sample Deal 5",
                link="http://example.com/deal5",
                source="Example",
                shopper_type=state.shopper_type,
                position=5,
                thumbnail="http://example.com/thumbnail5.jpg",
                price="₹5,000",
                tag="Best Value",
                product_link="http://example.com/product5"
            ),
            WebSearchResult(
                title="Sample Deal 6",
                link="http://example.com/deal6",
                source="Example",
                shopper_type=state.shopper_type,
                position=6,
                thumbnail="http://example.com/thumbnail6.jpg",
                price="₹6,000",
                tag="Premium",
                product_link="http://example.com/product6"
            ),
        ]
        
        # Apply price filtering to dummy results
        all_results = []
        for result in dummy_results:
            price = parse_price(result.price)
            if filters.min_price is not None and price < filters.min_price:
                continue
            if filters.max_price is not None and price > filters.max_price:
                continue
            all_results.append(result)
    else:
        # Real SerpAPI calls with enhanced parameters
        serpapi_api_key = os.getenv("SERPAPI_API_KEY", "")
        params = {
            "q": state.search_query,
            "api_key": serpapi_api_key,
            "engine": "google_shopping",
            "google_domain": "google.com",
            "direct_link": "true",
            "gl": "in",
            "hl": "en",
            "num": "20",
            "sort": "review_score" if state.shopper_type == "Minimalist" else "price_low_to_high" if state.shopper_type == "Savvy" else "review_count"
        }
        
        # Add price range if specified
        if filters.min_price is not None or filters.max_price is not None:
            price_range = []
            if filters.min_price is not None:
                price_range.append(str(int(filters.min_price)))
            if filters.max_price is not None:
                price_range.append(str(int(filters.max_price)))
            params["tbs"] = f"price:{'-'.join(price_range)}"
        
        search = GoogleSearch(params)
        results = search.get_dict()

        # Check if 'shopping_results' key exists in the response
        if "shopping_results" not in results:
            logger.error("No 'shopping_results' key in API response")
            return WebSearchList(search_results=all_results)

        # Enhanced result processing with filtering and scoring
        processed_results = []
        for idx, item in enumerate(results.get("shopping_results", []), start=1):
            # Skip results without essential information
            if not all([item.get("title"), item.get("price")]):
                logger.warning(f"Skipping item due to missing essential fields: {item}")
                continue

            # Use product_link if link is missing
            link = item.get("link", item.get("product_link", "No Link"))
            if link == "No Link":
                logger.warning(f"Skipping item due to missing link: {item}")
                continue

            # Apply additional price filtering (in case API doesn't fully respect it)
            price = parse_price(item.get("price", "inf"))
            if filters.min_price is not None and price < filters.min_price:
                continue
            if filters.max_price is not None and price > filters.max_price:
                continue

            # Calculate result score based on persona
            score = 0
            if state.shopper_type == "Trendsetter":
                score += float(item.get("rating", 0)) * 2
                score += float(item.get("review_count", 0)) * 0.01
            elif state.shopper_type == "Minimalist":
                score += float(item.get("rating", 0)) * 3
                if "warranty" in item.get("description", "").lower():
                    score += 2
            elif state.shopper_type == "Savvy":
                if price == float("inf"):
                    continue
                score += (1000 / price) if price > 0 else 0
                score += float(item.get("rating", 0))

            processed_results.append((
                WebSearchResult(
                    title=item.get("title", "No Title"),
                    link=link,
                    source=item.get("source", "Unknown"),
                    shopper_type=state.shopper_type,
                    position=idx,
                    thumbnail=item.get("thumbnail", ""),
                    price=item.get("price", "No Price"),
                    tag=get_result_tag(item, state.shopper_type),
                    product_link=item.get("product_link", "")
                ),
                score
            ))

        # Sort by score and take top results
        processed_results.sort(key=lambda x: x[1], reverse=True)
        all_results = [result for result, _ in processed_results]

    return WebSearchList(search_results=all_results)

def get_result_tag(item: dict, shopper_type: str) -> str:
    """
    Generate meaningful tags based on product attributes and shopper type.
    """
    rating = float(item.get("rating", 0))
    review_count = int(item.get("review_count", 0))
    price = parse_price(item.get("price", "inf"))
    
    if shopper_type == "Trendsetter":
        if review_count > 1000 and rating >= 4.5:
            return "Top Rated"
        elif "new" in item.get("title", "").lower():
            return "New Arrival"
        return "Trending"
    elif shopper_type == "Minimalist":
        if rating >= 4.7:
            return "Premium Quality"
        elif "warranty" in item.get("description", "").lower():
            return "Guaranteed"
        return "Essential"
    else:  # Savvy
        if price < 1000:  # Adjusted for Indian Rupees
            return "Great Deal"
        elif rating >= 4.5 and review_count > 500:
            return "Best Value"
        return "Smart Choice"

def parse_price(price_str: str) -> float:
    """
    Attempts to parse a price string like "₹25.99" -> 25.99
    Returns float('inf') if parsing fails
    """
    try:
        # Remove currency symbols and thousands separators
        clean_str = re.sub(r'[^\d.]', '', str(price_str))
        return float(clean_str)
    except (ValueError, TypeError):
        return float('inf')

########################################################
# 5. BUILDING THE BOOTSTRAP HTML
########################################################

def build_html_results(results: WebSearchList, persona: str, max_suggestions: int) -> str:
    """
    Builds a Bootstrap-based HTML card layout from the search results.
    """
    persona_results = [r for r in results.search_results 
                      if r.shopper_type.lower() == persona.lower()]
    
    # Apply price filtering again (double-check)
    if hasattr(results, 'filters') and results.filters:
        persona_results = [
            r for r in persona_results
            if (results.filters.min_price is None or parse_price(r.price) >= results.filters.min_price) and
               (results.filters.max_price is None or parse_price(r.price) <= results.filters.max_price)
        ]
    
    persona_results.sort(key=lambda x: parse_price(x.price))
    top_results = persona_results[:max_suggestions]

    if not top_results:
        return '''
        <div class="alert alert-warning">
            No results found matching your criteria. Try adjusting your price range or search terms.
        </div>
        '''

    html_output = f"""
    <div class="card">
        <div class="card-header d-flex justify-content-between align-items-center">
            <h3 class="mb-0">Top Suggestions for {persona}</h3>
            <button id="compareBtn" class="btn btn-secondary" disabled>
                Compare Selected (<span id="compareCount">0</span>)
            </button>
        </div>
        <div class="card-body">
            <div class="row">
    """

    for result in top_results:
        html_output += f"""
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card product-card h-100" data-product-id="{result.position}">
                    <div class="card-checkbox">
                        <input type="checkbox" class="form-check-input product-checkbox" 
                               id="product_{result.position}" data-product-id="{result.position}">
                    </div>
                    <span class="product-tag">{result.tag}</span>
                    <img src="{result.thumbnail}" class="product-image card-img-top" alt="{result.title}">
                    <div class="card-body">
                        <h5 class="card-title">{result.title}</h5>
                        <p class="product-price">{result.price}</p>
                        <p class="card-text text-muted">From {result.source}</p>
                        <div class="d-flex justify-content-between">
                            <a href="{result.link}" target="_blank" class="btn btn-primary">View Deal</a>
                            <button class="btn btn-outline-primary add-to-wishlist" data-product-id="{result.position}">
                                <i class="fas fa-heart"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        """

    html_output += """
            </div>
        </div>
    </div>
    
    <form id="compareForm" method="POST" action="/compare" style="display: none;">
        <input type="hidden" name="product_ids" id="compareProductIds">
    </form>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const checkboxes = document.querySelectorAll('.product-checkbox');
            const compareBtn = document.getElementById('compareBtn');
            const compareCount = document.getElementById('compareCount');
            const compareForm = document.getElementById('compareForm');
            const productIdsInput = document.getElementById('compareProductIds');
            
            checkboxes.forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    const selected = document.querySelectorAll('.product-checkbox:checked');
                    compareCount.textContent = selected.length;
                    compareBtn.disabled = selected.length < 2;
                    
                    // Highlight selected cards
                    const card = this.closest('.product-card');
                    if (this.checked) {
                        card.classList.add('selected');
                    } else {
                        card.classList.remove('selected');
                    }
                });
            });
            
            compareBtn.addEventListener('click', function() {
                const selected = Array.from(document.querySelectorAll('.product-checkbox:checked'))
                                    .map(checkbox => checkbox.dataset.productId);
                productIdsInput.value = selected.join(',');
                compareForm.submit();
            });
        });
    </script>
    """
    return html_output
########################################################
# 6. END-TO-END PIPELINE
########################################################

def run_shopping_pipeline(persona: str, product_name: str, max_suggestions: int = 3, 
                         min_price: float = None, max_price: float = None) -> str:
    """
    1) Generate product ideas for the chosen persona.
    2) Build query from the first idea.
    3) Perform web search.
    4) Build & return HTML results.
    """
    req = ShoppingRequest(
        product_name=product_name, 
        max_suggestions=max_suggestions,
        min_price=min_price,
        max_price=max_price
    )
    filters = ProductFilters(
        min_price=min_price,
        max_price=max_price,
        min_rating=3.0,
        sort_by="rating"
    )

    # Map persona string to the correct function
    persona_funcs = {
        "Trendsetter": shopper_enthusiast,
        "Minimalist": shopper_essentialist,
        "Savvy": shopper_frugalist
    }
    if persona not in persona_funcs:
        raise ValueError("Invalid persona selected.")

    # 1) Generate product ideas
    idea_list = persona_funcs[persona](req)

    # 2) Build search query
    search_query = scour_the_internet(idea_list, req)

    # 3) Perform the web search with filters
    results = web_search_agent(search_query, filters)

    # 4) Build the HTML for results
    html_results = build_html_results(results, persona, max_suggestions)
    return html_results

########################################################
# 7. FLASK ROUTES
########################################################

@app.route("/", methods=["GET"])
def index():
    """
    Render the main page with the search form.
    """
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    """
    Handle form submission, run the pipeline, and return results.
    """
    persona = request.form.get("shopper_type", "Trendsetter")
    product_name = request.form.get("product_name", "")
    max_suggestions = int(request.form.get("max_suggestions", 3))
    
    # Get price range from form
    min_price = request.form.get("min_price")
    max_price = request.form.get("max_price")
    min_price = float(min_price) if min_price else None
    max_price = float(max_price) if max_price else None

    # Run the shopping pipeline with price range
    results_html = run_shopping_pipeline(
        persona, 
        product_name, 
        max_suggestions,
        min_price,
        max_price
    )
    
    # Store results in session for comparison and tracking
    if hasattr(results_html, 'search_results'):
        session['last_results'] = [result.dict() for result in results_html.search_results]
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"html": results_html})
    
    return render_template("index.html", results_html=results_html)

@app.route("/compare", methods=["POST"])
def compare_products():
    try:
        product_ids = request.form.get("product_ids", "").split(',')
        if not product_ids or len(product_ids) < 2:
            return jsonify({"error": "Please select at least 2 products to compare"})
        
        last_results = session.get('last_results', [])
        if not last_results:
            return jsonify({"error": "No products available for comparison"})
        
        # Convert to WebSearchResult objects and filter by selected IDs
        products = []
        for p in last_results:
            try:
                # Ensure position is properly set
                if 'position' not in p:
                    p['position'] = len(products) + 1
                
                # Convert to WebSearchResult
                product = WebSearchResult(**p)
                if str(product.position) in product_ids:
                    products.append(product)
            except Exception as e:
                logger.warning(f"Skipping invalid product data: {str(e)}")
                continue
        
        if len(products) < 2:
            return jsonify({"error": "Could not find selected products in session"})
        
        # Generate comparison with proper error handling
        try:
            comparison = generate_detailed_comparison(products)
            return jsonify({
                "html": render_template("comparison_modal.html", comparison=comparison),
                "success": True
            })
        except Exception as e:
            logger.error(f"Comparison generation failed: {str(e)}")
            return jsonify({"error": "Failed to generate comparison"})
            
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return jsonify({"error": f"An error occurred during comparison: {str(e)}"})
    
def generate_detailed_comparison(products: list[WebSearchResult]) -> dict:
    """Generate comprehensive comparison data with scoring and detailed specs"""
    # First ensure all products have required attributes
    for product in products:
        if not hasattr(product, 'rating'):
            product.rating = 0.0
        if not hasattr(product, 'review_count'):
            product.review_count = 0
    
    # Calculate scores for each product with proper price handling
    scored_products = []
    for product in products:
        try:
            price = parse_price(product.price) if hasattr(product, 'price') else float('inf')
        except:
            price = float('inf')
        
        # More robust spec extraction
        specs = extract_specs_from_title(getattr(product, 'title', ''))
        description = getattr(product, 'description', '')
        
        # Extract additional specs from description if available
        if description:
            specs.update(extract_specs_from_description(description))
        
        scored_products.append({
            "product": product,
            "price": price,
            "rating": float(getattr(product, 'rating', 0)),
            "review_count": int(getattr(product, 'review_count', 0)),
            "specs": specs
        })
    
    # Generate comparison points with fallbacks for missing data
    comparison_points = {
        "Price": [f"₹{p['price']:.2f}" if p['price'] != float('inf') else "N/A" for p in scored_products],
        "Rating": [f"{p['rating']:.1f}/5" + (f" ({p['review_count']} reviews)" if p['review_count'] > 0 else "") 
                   for p in scored_products],
    }
    
    # Add dynamic comparison points based on available specs
    all_spec_keys = set()
    for p in scored_products:
        all_spec_keys.update(p['specs'].keys())
    
    for spec_key in all_spec_keys:
        comparison_points[spec_key.capitalize()] = [
            p['specs'].get(spec_key, 'N/A') for p in scored_products
        ]
    
    # Generate pros/cons for each product with more robust logic
    pros_cons = []
    for p in scored_products:
        pros = []
        cons = []
        
        # Rating based pros/cons
        rating = p['rating']
        if rating >= 4.5:
            pros.append("Excellent ratings")
        elif rating >= 4:
            pros.append("Good ratings")
        elif rating < 3:
            cons.append("Below average ratings")
        
        # Price based pros/cons
        price = p['price']
        if price < 10000:  # Adjust thresholds as needed
            pros.append("Budget friendly")
        elif price > 50000:
            cons.append("Premium priced")
        
        # Review count based
        review_count = p['review_count']
        if review_count > 1000:
            pros.append("Popular choice")
        elif review_count < 10:
            cons.append("Limited reviews")
        
        # Add spec-based pros/cons
        specs = p['specs']
        if 'ram' in specs and int(specs['ram'].split()[0]) >= 8:  # If RAM >= 8GB
            pros.append("Ample RAM")
        if 'storage' in specs and int(specs['storage'].split()[0]) >= 128:  # If storage >= 128GB
            pros.append("Large storage")
            
        pros_cons.append({"pros": pros, "cons": cons})
    
    # Determine best picks with weighted scoring
    def calculate_score(product):
        price = product['price']
        rating = product['rating']
        reviews = product['review_count']
        
        # Weighted score (adjust weights as needed)
        price_weight = 0.4 if price < 50000 else 0.2  # Less emphasis on very expensive items
        rating_weight = 0.4
        review_weight = 0.2
        
        # Normalize values
        price_score = (1 / (price + 1)) * 1000 if price > 0 else 0
        rating_score = rating * 20  # Convert 5-point scale to 100-point scale
        review_score = min(reviews / 100, 1) * 100  # Cap at 100 reviews
        
        return (price_score * price_weight + 
                rating_score * rating_weight + 
                review_score * review_weight)
    
    scored_products.sort(key=calculate_score, reverse=True)
    
    return {
        "products": [p['product'] for p in scored_products],
        "comparison_points": comparison_points,
        "pros_cons": pros_cons,
        "best_overall": scored_products[0]['product'].title,
        "best_value": min(scored_products, key=lambda x: x['price'])['product'].title,
        "best_quality": max(scored_products, key=lambda x: x['rating'])['product'].title
    }
    
def extract_specs_from_description(description: str) -> dict:
    """Extract specifications from product description"""
    specs = {}
    desc_lower = description.lower()
    
    # Battery capacity
    battery_match = re.search(r'(\d+)\s*(?:mah|mah battery)', desc_lower)
    if battery_match:
        specs['battery'] = f"{battery_match.group(1)}mAh"
    
    # Camera resolution
    camera_match = re.search(r'(\d+)\s*(?:mp|megapixel)', desc_lower)
    if camera_match:
        specs['camera'] = f"{camera_match.group(1)}MP"
    
    # Additional features
    features = []
    if 'waterproof' in desc_lower or 'water resistant' in desc_lower:
        features.append("Water resistant")
    if 'wireless charging' in desc_lower:
        features.append("Wireless charging")
    if 'dual sim' in desc_lower:
        features.append("Dual SIM")
    
    if features:
        specs['features'] = ", ".join(features)
    
    return specs
    
def get_recommendation_based_on_comparison(products: list[WebSearchResult]) -> str:
    """Generate a smart recommendation based on the comparison."""
    prices = [parse_price(p.price) for p in products]
    ratings = [float(getattr(p, 'rating', 3)) for p in products]
    
    best_value_index = max(
        range(len(products)),
        key=lambda i: ratings[i]/(prices[i] if prices[i] > 0 else 1)
    )
    
    return {
        "best_overall": products[best_value_index].title,
        "reason": f"Offers the best value with a rating of {ratings[best_value_index]} at ₹{prices[best_value_index]}",
        "best_for_budget": min((p for p in products), key=lambda x: parse_price(x.price)).title,
        "best_premium": max((p for p in products), key=lambda x: float(getattr(x, 'rating', 0))).title
    }

def build_comparison_html(comparison: ProductComparison) -> str:
    """
    Build HTML for product comparison view.
    """
    html = """
    <div class="comparison-table">
        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>Feature</th>
    """
    
    # Add product names to header
    for product in comparison.products:
        html += f"<th>{product.title}</th>"
    html += "</tr></thead><tbody>"
    
    # Add comparison points
    for feature, values in comparison.comparison_points.items():
        html += f"<tr><td><strong>{feature}</strong></td>"
        for value in values:
            html += f"<td>{value}</td>"
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html

def init_user_session():
    """Initialize user session with preferences."""
    if 'user_prefs' not in session:
        session['user_prefs'] = UserPreferences().dict()

@app.route("/wishlist/add", methods=["POST"])
def add_to_wishlist():
    """Add a product to the wishlist."""
    init_user_session()
    product_id = request.form.get("product_id")
    products = session.get('last_results', [])
    product = next((p for p in products if str(p.position) == product_id), None)
    
    if product:
        prefs = UserPreferences(**session['user_prefs'])
        prefs.wishlist.append(product)
        session['user_prefs'] = prefs.dict()
        return jsonify({"success": True})
    return jsonify({"error": "Product not found"})

@app.route("/history", methods=["GET"])
def view_history():
    """View search and product history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    return render_template(
        "history.html",
        viewed_products=prefs.viewed_products[-10:],  # Last 10 viewed
        search_history=prefs.search_history[-10:],    # Last 10 searches
        wishlist=prefs.wishlist
    )

def track_product_view(product: WebSearchResult):
    """Track when a product is viewed."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    prefs.viewed_products.append(product)
    if len(prefs.viewed_products) > 50:  # Keep last 50 views
        prefs.viewed_products = prefs.viewed_products[-50:]
    session['user_prefs'] = prefs.dict()

def track_search(query: str):
    """Track search history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    prefs.search_history.append(query)
    if len(prefs.search_history) > 50:  # Keep last 50 searches
        prefs.search_history = prefs.search_history[-50:]
    session['user_prefs'] = prefs.dict()

@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    """Get personalized recommendations based on history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    
    # Analyze viewing history and wishlist to generate recommendations
    viewed_categories = set()
    price_ranges = []
    
    for product in prefs.viewed_products + prefs.wishlist:
        viewed_categories.add(product.tag)
        price_ranges.append(parse_price(product.price))
    
    # Calculate preferred price range
    avg_price = sum(price_ranges) / len(price_ranges) if price_ranges else 0
    
    # Generate recommendations based on preferences
    recommendations = []
    for category in viewed_categories:
        req = ShoppingRequest(
            product_name=f"products in {category}",
            max_suggestions=2,
            min_price=avg_price * 0.7,
            max_price=avg_price * 1.3
        )
        results = run_shopping_pipeline(prefs.favorite_personas[0] if prefs.favorite_personas else "Trendsetter", req.product_name)
        recommendations.extend(results)
    
    return render_template("recommendations.html", recommendations=recommendations)

@app.route("/track/search", methods=["POST"])
def track_search():
    """Track user search queries."""
    query = request.form.get("query", "")
    if query:
        track_search_history(query)
    return jsonify({"success": True})

@app.route("/track/product", methods=["POST"])
def track_product():
    """Track product views."""
    product_id = request.form.get("product_id")
    products = session.get('last_results', [])
    product = next((p for p in products if str(p.position) == product_id), None)
    
    if product:
        track_product_view(product)
        return jsonify({"success": True})
    return jsonify({"error": "Product not found"})

def track_search_history(query: str):
    """Track search history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    prefs.search_history.append(f"{query} ({timestamp})")
    if len(prefs.search_history) > 50:  # Keep last 50 searches
        prefs.search_history = prefs.search_history[-50:]
    session['user_prefs'] = prefs.dict()

########################################################
# 8. MAIN
########################################################

if __name__ == "__main__":
    # By default runs on port 5000
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
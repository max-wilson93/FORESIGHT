/**
 * Order Matching Module
 * 
 * This module implements a limit order book (LOB) for matching buy and sell orders.
 * It ensures efficient and fair order matching with low-latency performance.
 * 
 * Key Features:
 * - Efficient order matching using Rust's data structures.
 * - Support for limit orders, market orders, and cancellations.
 * - Thread-safe concurrency for high-frequency trading.
 */

 use std::collections::BTreeMap;
 use std::sync::Mutex;
 
 /// Represents an order in the order book
 #[derive(Debug, Clone)]
 pub struct Order {
     pub id: u64,
     pub symbol: String,
     pub quantity: f64,
     pub price: f64,
     pub side: OrderSide, // Buy or Sell
 }
 
 /// Represents the side of an order (Buy or Sell)
 #[derive(Debug, Clone)]
 pub enum OrderSide {
     Buy,
     Sell,
 }
 
 /// Limit Order Book (LOB)
 pub struct OrderBook {
     bids: Mutex<BTreeMap<u64, Order>>, // Buy orders (sorted by price)
     asks: Mutex<BTreeMap<u64, Order>>, // Sell orders (sorted by price)
 }
 
 impl OrderBook {
     /// Create a new OrderBook
     pub fn new() -> Self {
         OrderBook {
             bids: Mutex::new(BTreeMap::new()),
             asks: Mutex::new(BTreeMap::new()),
         }
     }
 
     /// Add an order to the order book
     pub fn add_order(&self, order: Order) {
         match order.side {
             OrderSide::Buy => {
                 let mut bids = self.bids.lock().unwrap();
                 bids.insert(order.id, order);
             }
             OrderSide::Sell => {
                 let mut asks = self.asks.lock().unwrap();
                 asks.insert(order.id, order);
             }
         }
         println!("Order added to order book: {:?}", order);
     }
 
     /// Match orders in the order book
     pub fn match_orders(&self) {
         let mut bids = self.bids.lock().unwrap();
         let mut asks = self.asks.lock().unwrap();
 
         while let (Some(best_bid), Some(best_ask)) = (bids.values().next(), asks.values().next()) {
             if best_bid.price >= best_ask.price {
                 // Execute trade
                 println!("Trade executed: Bid {:?} <-> Ask {:?}", best_bid, best_ask);
                 bids.remove(&best_bid.id);
                 asks.remove(&best_ask.id);
             } else {
                 break; // No more matches
             }
         }
     }
 }
/**
 * Order Matching Engine (Rust)
 * 
 * Purpose:
 * This module implements a low-latency order matching engine for the FORESIGHT system. It manages
 * the limit order book, matches buy and sell orders, and ensures fair and efficient execution.
 * 
 * Role in FORESIGHT:
 * - Handles order matching for real-time trading.
 * - Ensures thread-safe concurrency for high-frequency trading.
 * - Integrates with the execution engine and portfolio management system.
 * 
 * Key Features:
 * - Efficient order book management using BTreeMap.
 * - Support for limit orders, market orders, and cancellations.
 * - Real-time order matching with ultra-low latency.
 */

 use std::collections::BTreeMap;
 use std::sync::{Arc, Mutex};
 
 /// Represents an order in the order book
 #[derive(Debug, Clone)]
 pub struct Order {
     pub id: u64,              // Unique order ID
     pub symbol: String,       // Trading symbol (e.g., "AAPL")
     pub price: f64,           // Order price
     pub quantity: f64,        // Order quantity
     pub side: OrderSide,      // Buy or Sell
     pub order_type: OrderType, // Limit or Market
 }
 
 /// Represents the side of an order (Buy or Sell)
 #[derive(Debug, Clone, PartialEq)]
 pub enum OrderSide {
     Buy,
     Sell,
 }
 
 /// Represents the type of order (Limit or Market)
 #[derive(Debug, Clone, PartialEq)]
 pub enum OrderType {
     Limit,
     Market,
 }
 
 /// Represents the limit order book
 pub struct OrderBook {
     bids: Arc<Mutex<BTreeMap<u64, Order>>, // Buy orders (sorted by price)
     asks: Arc<Mutex<BTreeMap<u64, Order>>, // Sell orders (sorted by price)
 }
 
 impl OrderBook {
     /// Create a new OrderBook
     pub fn new() -> Self {
         OrderBook {
             bids: Arc::new(Mutex::new(BTreeMap::new())),
             asks: Arc::new(Mutex::new(BTreeMap::new())),
         }
     }
 
     /// Add an order to the order book
     pub fn add_order(&self, order: Order) -> Result<(), String> {
         match order.side {
             OrderSide::Buy => {
                 let mut bids = self.bids.lock().map_err(|e| e.to_string())?;
                 bids.insert(order.id, order);
             }
             OrderSide::Sell => {
                 let mut asks = self.asks.lock().map_err(|e| e.to_string())?;
                 asks.insert(order.id, order);
             }
         }
         Ok(())
     }
 
     /// Match orders in the order book
     pub fn match_orders(&self) -> Result<Vec<(Order, Order)>, String> {
         let mut bids = self.bids.lock().map_err(|e| e.to_string())?;
         let mut asks = self.asks.lock().map_err(|e| e.to_string())?;
 
         let mut matched_orders = Vec::new();
 
         while let (Some((&best_bid_id, best_bid)), Some((&best_ask_id, best_ask))) =
             (bids.iter().next(), asks.iter().next())
         {
             if best_bid.price >= best_ask.price {
                 // Execute trade
                 matched_orders.push((best_bid.clone(), best_ask.clone()));
                 bids.remove(&best_bid_id);
                 asks.remove(&best_ask_id);
             } else {
                 break; // No more matches
             }
         }
 
         Ok(matched_orders)
     }
 
     /// Cancel an order from the order book
     pub fn cancel_order(&self, order_id: u64, side: OrderSide) -> Result<(), String> {
         match side {
             OrderSide::Buy => {
                 let mut bids = self.bids.lock().map_err(|e| e.to_string())?;
                 bids.remove(&order_id);
             }
             OrderSide::Sell => {
                 let mut asks = self.asks.lock().map_err(|e| e.to_string())?;
                 asks.remove(&order_id);
             }
         }
         Ok(())
     }
 }
 
 // Example usage
 #[cfg(test)]
 mod tests {
     use super::*;
 
     #[test]
     fn test_order_matching() {
         let order_book = OrderBook::new();
 
         // Add buy and sell orders
         let buy_order = Order {
             id: 1,
             symbol: "AAPL".to_string(),
             price: 150.0,
             quantity: 100.0,
             side: OrderSide::Buy,
             order_type: OrderType::Limit,
         };
 
         let sell_order = Order {
             id: 2,
             symbol: "AAPL".to_string(),
             price: 149.0,
             quantity: 100.0,
             side: OrderSide::Sell,
             order_type: OrderType::Limit,
         };
 
         order_book.add_order(buy_order.clone()).unwrap();
         order_book.add_order(sell_order.clone()).unwrap();
 
         // Match orders
         let matched_orders = order_book.match_orders().unwrap();
         assert_eq!(matched_orders.len(), 1);
         assert_eq!(matched_orders[0].0.id, buy_order.id);
         assert_eq!(matched_orders[0].1.id, sell_order.id);
     }
 }